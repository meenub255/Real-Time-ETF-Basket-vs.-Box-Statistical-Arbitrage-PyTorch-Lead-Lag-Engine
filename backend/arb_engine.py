"""
Basket vs. Box Arbitrage Pricing Engine, 3D Price Gap Surface Generator,
and Institutional Trade Ticket / Execution Blotter.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Dict, List

import numpy as np

from backend.lead_lag_model import LeadLagPrediction


class ArbitrageEngine:
    """
    Computes the live Arbitrage Trade Ticket (exact basket of stocks to buy/sell,
    gross-to-net profit after fees), maintains the 3D Price Gap Surface matrices,
    and manages manual + automated arbitrage executions.
    """

    def __init__(self, default_lot_size: int = 1000):
        self.qqq_lot_size: int = default_lot_size
        self.min_net_profit_threshold_usd: float = 45.0  # Min net profit for executable flag
        self.auto_arb_enabled: bool = False
        self.last_auto_exec_ts: float = 0.0

        # Rolling surface history: stores 24 recent slices of gap across lag offsets
        self.lag_offsets_ms: List[int] = [-200, -150, -100, -50, 0, 50, 100, 150, 200, 250]
        self.surface_z_history: deque = deque(maxlen=24)
        self.surface_time_labels: deque = deque(maxlen=24)

        # Initialize surface with smooth baseline
        for i in range(24):
            self.surface_z_history.append([0.0 for _ in self.lag_offsets_ms])
            self.surface_time_labels.append(f"T-{24 - i}")

        # Rolling statistical z-score buffer for the dollar price gap
        self.gap_usd_history: deque = deque(maxlen=120)
        self.gap_usd_history.extend([0.02, -0.03, 0.05, -0.04, 0.01, 0.06, -0.02, 0.04])

        # Executed Arbitrage Trade Blotter
        self.executed_trades: deque = deque(maxlen=50)
        self.cumulative_net_pnl_usd: float = 0.0
        self.cumulative_gross_pnl_usd: float = 0.0
        self.total_boxes_executed: int = 0

    def set_config(
        self,
        qqq_lot_size: int | None = None,
        min_net_profit_threshold_usd: float | None = None,
        auto_arb_enabled: bool | None = None,
    ) -> Dict[str, object]:
        if qqq_lot_size is not None:
            self.qqq_lot_size = max(10, min(100000, int(qqq_lot_size)))
        if min_net_profit_threshold_usd is not None:
            self.min_net_profit_threshold_usd = max(1.0, float(min_net_profit_threshold_usd))
        if auto_arb_enabled is not None:
            self.auto_arb_enabled = bool(auto_arb_enabled)
        return {
            "qqq_lot_size": self.qqq_lot_size,
            "min_net_profit_threshold_usd": self.min_net_profit_threshold_usd,
            "auto_arb_enabled": self.auto_arb_enabled,
        }

    def evaluate_tick(
        self,
        market_snapshot: Dict[str, object],
        prediction: LeadLagPrediction,
    ) -> Dict[str, object]:
        """
        Evaluates the current Basket vs. Box arbitrage state, updates the 3D Price Gap
        Surface, builds the live Trade Ticket, and triggers Auto-Arb if enabled.
        """
        qqq_price = float(market_snapshot["qqq"]["price"])
        box_price = float(market_snapshot["box"]["price"])
        gap_usd = float(market_snapshot["spread"]["gap_usd"])
        gap_bps = float(market_snapshot["spread"]["gap_bps"])
        constituents = market_snapshot["constituents"]
        tick_count = int(market_snapshot["tick_count"])

        self.gap_usd_history.append(gap_usd)
        gap_arr = np.array(self.gap_usd_history, dtype=np.float64)
        gap_std = float(max(np.std(gap_arr), 0.045))
        gap_mean = float(np.mean(gap_arr))
        z_score = float((gap_usd - gap_mean) / gap_std)

        # 1. Update 3D Price Gap Surface
        # Across lag offsets tau in [-200ms ... +250ms], compute the projected price gap ($)
        # incorporating the PyTorch Lead-Lag decay curve and NVDA impulse wave
        nvda_imp_usd = prediction.qqq_unreacted_gap_usd
        pred_delta_usd = prediction.predicted_qqq_delta_usd

        z_row: List[float] = []
        for tau_ms in self.lag_offsets_ms:
            if tau_ms == 0:
                val = gap_usd
            elif tau_ms > 0:
                # Forward lead-lag projection: how the gap evolves over the next tau_ms
                decay = np.exp(-tau_ms / 180.0)
                lead_wave = np.sin(tau_ms / 95.0) * (pred_delta_usd * 0.85 - nvda_imp_usd * 0.6)
                val = gap_usd * decay + lead_wave
            else:
                # Historical lag coupling with NVDA shock wave
                past_decay = np.exp(tau_ms / 160.0)
                val = gap_usd * past_decay + nvda_imp_usd * np.cos(tau_ms / 80.0)
            z_row.append(round(float(val), 4))

        self.surface_z_history.append(z_row)
        self.surface_time_labels.append(f"t-{tick_count}")

        # Also compute a Parametric 3D Surface: NVDA Shock (%) x Reaction Lag (ms) -> Price Gap ($)
        nvda_shocks_pct = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        reaction_lags_ms = [25, 50, 100, 150, 200, 250]
        parametric_z: List[List[float]] = []
        for lag in reaction_lags_ms:
            row_p = []
            unreacted_frac = float(np.exp(-lag / 140.0))
            for shock_pct in nvda_shocks_pct:
                # NVDA represents ~24.8% of the Top-5 normalized box
                box_shift_usd = box_price * (shock_pct / 100.0) * 0.248
                # If NVDA jumps +1.5%, Box rises immediately while QQQ lags -> negative gap (QQQ - Box)
                effective_gap = gap_usd - box_shift_usd * unreacted_frac
                row_p.append(round(float(effective_gap), 4))
            parametric_z.append(row_p)

        # 2. Construct Live Arbitrage Trade Ticket
        lot = self.qqq_lot_size

        # Determine direction:
        # Case A: gap_usd >= 0 (e.g. QQQ = $500.45, Box = $500.00 -> Gap = +$0.45)
        #         QQQ is OVERPRICED relative to the stocks inside it!
        #         Jane Street Trade: SELL Overpriced QQQ ETF & BUY Cheaper Stock Basket
        # Case B: gap_usd < 0 (e.g. QQQ = $499.55, Box = $500.00 -> Gap = -$0.45)
        #         QQQ is UNDERPRICED relative to the stocks inside it!
        #         Jane Street Trade: BUY Cheap QQQ ETF & SHORT Overpriced Stock Basket
        if gap_usd >= 0.0:
            strategy_type = "SHORT_QQQ_LONG_BASKET"
            headline = "SELL OVERPRICED QQQ / BUY CHEAPER STOCK BASKET"
            qqq_action = "SELL SHORT"
            basket_action = "BUY"
        else:
            strategy_type = "LONG_QQQ_SHORT_BASKET"
            headline = "BUY UNDERPRICED QQQ / SELL OVERPRICED STOCK BASKET"
            qqq_action = "BUY"
            basket_action = "SELL SHORT"

        abs_gap_per_share = abs(gap_usd)
        gross_spread_profit_usd = abs_gap_per_share * lot

        # Lead-Lag PyTorch Alpha Contribution:
        # If we are buying QQQ (gap < 0) and PyTorch predicts QQQ is about to jump up (pred_delta_usd > 0),
        # or if we are selling QQQ (gap > 0) and PyTorch predicts QQQ is about to drop (pred_delta_usd < 0),
        # the Lead-Lag predictor provides additional millisecond front-running alpha!
        if (gap_usd < 0 and prediction.predicted_qqq_delta_usd > 0) or (
            gap_usd >= 0 and prediction.predicted_qqq_delta_usd < 0
        ):
            lead_lag_alpha_per_share = abs(prediction.predicted_qqq_delta_usd) * 0.65
        else:
            lead_lag_alpha_per_share = abs(prediction.predicted_qqq_delta_usd) * 0.15

        lead_lag_alpha_usd = lead_lag_alpha_per_share * lot

        # Institutional Cost & Fee Model (Jane Street / Direct Market Access tier)
        notional_qqq_usd = qqq_price * lot
        notional_box_usd = box_price * lot

        # 1) Bid/Ask Midpoint Peg Slippage (~0.28 bps across ETF + Basket)
        slippage_cost_usd = (notional_qqq_usd + notional_box_usd) * 0.000028
        # 2) Exchange Make/Take + SEC Section 31 + FINRA TAF Fees (~0.09 bps)
        exchange_reg_fees_usd = (notional_qqq_usd + notional_box_usd) * 0.000009
        # 3) Intraday Prime Broker Borrow / Clearing Fee (~0.04 bps)
        borrow_clearing_usd = notional_qqq_usd * 0.000004

        total_fees_usd = slippage_cost_usd + exchange_reg_fees_usd + borrow_clearing_usd
        net_profit_usd = gross_spread_profit_usd + lead_lag_alpha_usd - total_fees_usd
        net_profit_per_share = net_profit_usd / max(lot, 1)
        net_margin_bps = (net_profit_usd / max(notional_box_usd, 1.0)) * 10000.0

        # Build exact multi-leg basket order table
        basket_legs: List[Dict[str, object]] = []
        for c in constituents:
            exact_shares = float(c["shares_per_qqq"]) * lot
            whole_shares = int(round(exact_shares))
            leg_notional = exact_shares * float(c["price"])
            basket_legs.append(
                {
                    "symbol": c["symbol"],
                    "name": c["name"],
                    "side": basket_action,
                    "price": c["price"],
                    "bid": c["bid"],
                    "ask": c["ask"],
                    "exact_shares": round(exact_shares, 2),
                    "whole_shares": whole_shares,
                    "notional_usd": round(leg_notional, 2),
                    "box_weight_pct": c["box_norm_weight_pct"],
                    "qqq_raw_weight_pct": c["qqq_weight_pct"],
                }
            )

        is_executable = net_profit_usd >= self.min_net_profit_threshold_usd

        ticket = {
            "strategy_type": strategy_type,
            "headline": headline,
            "is_executable": is_executable,
            "status_badge": (
                "LOCKED ARBITRAGE OPPORTUNITY"
                if is_executable
                else "SPREAD INSIDE FEE THRESHOLD"
            ),
            "qqq_lot_size": lot,
            "qqq_leg": {
                "symbol": "QQQ",
                "side": qqq_action,
                "shares": lot,
                "price": qqq_price,
                "notional_usd": round(notional_qqq_usd, 2),
            },
            "box_summary": {
                "side": basket_action,
                "implied_price_per_qqq": box_price,
                "total_notional_usd": round(notional_box_usd, 2),
            },
            "basket_legs": basket_legs,
            "economics": {
                "gap_per_share_usd": round(gap_usd, 4),
                "abs_gap_per_share_usd": round(abs_gap_per_share, 4),
                "gross_spread_profit_usd": round(gross_spread_profit_usd, 2),
                "lead_lag_alpha_usd": round(lead_lag_alpha_usd, 2),
                "slippage_cost_usd": round(slippage_cost_usd, 2),
                "exchange_reg_fees_usd": round(exchange_reg_fees_usd, 2),
                "borrow_clearing_usd": round(borrow_clearing_usd, 2),
                "total_fees_usd": round(total_fees_usd, 2),
                "net_profit_usd": round(net_profit_usd, 2),
                "net_profit_per_share_usd": round(net_profit_per_share, 4),
                "net_margin_bps": round(net_margin_bps, 2),
            },
        }

        # Check Auto-Arbitrage execution trigger (cooldown 1.2s between auto-fills)
        now = time.time()
        auto_triggered_trade = None
        if (
            self.auto_arb_enabled
            and is_executable
            and (now - self.last_auto_exec_ts) >= 1.2
        ):
            auto_triggered_trade = self.execute_ticket(ticket, trigger_source="AUTO_ARB_BOT")
            self.last_auto_exec_ts = now

        return {
            "z_score": round(z_score, 2),
            "gap_std_usd": round(gap_std, 4),
            "surface_3d": {
                "x_lags_ms": self.lag_offsets_ms,
                "y_time_labels": list(self.surface_time_labels),
                "z_gap_matrix": list(self.surface_z_history),
                "parametric": {
                    "x_nvda_shocks_pct": nvda_shocks_pct,
                    "y_reaction_lags_ms": reaction_lags_ms,
                    "z_gap_matrix": parametric_z,
                },
            },
            "ticket": ticket,
            "auto_triggered_trade": auto_triggered_trade,
            "blotter_summary": self.get_blotter_summary(),
        }

    def execute_ticket(
        self, ticket: Dict[str, object], trigger_source: str = "MANUAL_TRADER"
    ) -> Dict[str, object]:
        """
        Records an atomic multi-leg Basket vs. Box arbitrage execution in the blotter.
        """
        econ = ticket["economics"]
        qqq_leg = ticket["qqq_leg"]
        box_sum = ticket["box_summary"]

        net_pnl = float(econ["net_profit_usd"])
        gross_pnl = float(econ["gross_spread_profit_usd"]) + float(econ["lead_lag_alpha_usd"])

        self.cumulative_net_pnl_usd += net_pnl
        self.cumulative_gross_pnl_usd += gross_pnl
        self.total_boxes_executed += 1

        fill_record = {
            "trade_id": f"ARB-{str(uuid.uuid4())[:6].upper()}",
            "timestamp": time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}",
            "trigger_source": trigger_source,
            "headline": ticket["headline"],
            "qqq_side": qqq_leg["side"],
            "qqq_shares": qqq_leg["shares"],
            "qqq_price": qqq_leg["price"],
            "box_price": box_sum["implied_price_per_qqq"],
            "gap_per_share_usd": econ["gap_per_share_usd"],
            "gross_profit_usd": round(gross_pnl, 2),
            "fees_usd": econ["total_fees_usd"],
            "net_profit_usd": round(net_pnl, 2),
            "fill_latency_us": int(np.random.randint(115, 240)),
            "legs_summary": ", ".join(
                f"{leg['side'].split()[0]} {leg['whole_shares']} {leg['symbol']} @ ${leg['price']:.2f}"
                for leg in ticket["basket_legs"]
            ),
        }
        self.executed_trades.appendleft(fill_record)
        return fill_record

    def get_blotter_summary(self) -> Dict[str, object]:
        return {
            "cumulative_net_pnl_usd": round(self.cumulative_net_pnl_usd, 2),
            "cumulative_gross_pnl_usd": round(self.cumulative_gross_pnl_usd, 2),
            "total_boxes_executed": self.total_boxes_executed,
            "auto_arb_enabled": self.auto_arb_enabled,
            "qqq_lot_size": self.qqq_lot_size,
            "min_net_profit_threshold_usd": self.min_net_profit_threshold_usd,
            "recent_trades": list(self.executed_trades)[:15],
        }
