"""
Hybrid Market Data & Microstructure Lead-Lag Simulation Engine.

Combines:
1. Live baseline price discovery via Yahoo Finance API (with graceful fallback to
   calibrated Nasdaq-100 / QQQ reference prices around $500.00).
2. Sub-second (50ms-80ms) tick-by-tick microstructure simulation with realistic
   lead-lag propagation (NVDA & mega-caps lead QQQ by 100-200ms during order-flow bursts)
   and Ornstein-Uhlenbeck ETF-vs-Box basis dislocation dynamics.
3. Interactive Market Shock Injectors (including the textbook "$500.45 QQQ vs. $500.00 Box"
   Jane Street arbitrage scenario, NVDA momentum breakouts, and flash sweeps).
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ConstituentConfig:
    symbol: str
    name: str
    qqq_weight: float         # Actual weight in QQQ ETF (~35.97% combined for Top 5)
    box_norm_weight: float    # Normalized weight inside the 5-stock Replication Box (sums to 1.0)
    default_price: float
    half_spread_bps: float    # Typical institutional bid-ask half-spread in bps
    volatility_bps: float     # Per-tick microstructure volatility in bps


CONSTITUENTS: Dict[str, ConstituentConfig] = {
    "NVDA": ConstituentConfig(
        symbol="NVDA",
        name="NVIDIA Corp.",
        qqq_weight=0.0892,
        box_norm_weight=0.2480,
        default_price=138.50,
        half_spread_bps=0.85,
        volatility_bps=2.40,
    ),
    "AAPL": ConstituentConfig(
        symbol="AAPL",
        name="Apple Inc.",
        qqq_weight=0.0868,
        box_norm_weight=0.2413,
        default_price=228.00,
        half_spread_bps=0.65,
        volatility_bps=1.45,
    ),
    "MSFT": ConstituentConfig(
        symbol="MSFT",
        name="Microsoft Corp.",
        qqq_weight=0.0812,
        box_norm_weight=0.2257,
        default_price=425.00,
        half_spread_bps=0.65,
        volatility_bps=1.40,
    ),
    "AMZN": ConstituentConfig(
        symbol="AMZN",
        name="Amazon.com Inc.",
        qqq_weight=0.0538,
        box_norm_weight=0.1496,
        default_price=210.00,
        half_spread_bps=0.75,
        volatility_bps=1.65,
    ),
    "META": ConstituentConfig(
        symbol="META",
        name="Meta Platforms",
        qqq_weight=0.0487,
        box_norm_weight=0.1354,
        default_price=590.00,
        half_spread_bps=0.80,
        volatility_bps=1.75,
    ),
}


class HybridMarketEngine:
    """
    Maintains real-time prices, order-book bid/ask quotes, constituent multipliers,
    and lead-lag propagation buffers for QQQ and its top 5 constituents.
    """

    def __init__(self, tick_interval_ms: int = 60):
        self.tick_interval_ms = tick_interval_ms
        self.rng = np.random.default_rng(101)
        self.tick_count: int = 0
        self.data_source_mode: str = "HYBRID_LIVE_SIM"
        self.last_live_sync_ts: Optional[float] = None
        self.active_shock_label: str = "Normal Microstructure Flow"
        self.shock_Banner_ttl: int = 0

        # Anchor baseline prices (calibrated so Synthetic Box = $500.00 at equilibrium)
        self.anchor_qqq_price: float = 500.00
        self.anchor_stock_prices: Dict[str, float] = {
            sym: cfg.default_price for sym, cfg in CONSTITUENTS.items()
        }

        # Live evolving state
        self.stock_prices: Dict[str, float] = dict(self.anchor_stock_prices)
        self.prev_stock_prices: Dict[str, float] = dict(self.anchor_stock_prices)
        self.qqq_price: float = 500.00
        self.prev_qqq_price: float = 500.00
        self.residual_index_factor: float = 1.0000  # Remaining 95 stocks factor

        # Share multipliers per 1 unit of Synthetic QQQ Box so sum(shares_i * P_i_0) == QQQ_0
        self.box_share_weights: Dict[str, float] = {}
        self._recalibrate_box_multipliers()

        # Queue of delayed constituent impulses that have not yet propagated into QQQ
        # Each item: (ticks_until_qqq_reaction, delta_qqq_bps_impulse)
        self.propagation_queue: deque = deque(maxlen=16)
        for _ in range(6):
            self.propagation_queue.append(0.0)

        # Explicitly injected ETF mispricing offset (in USD) that mean-reverts via OU process
        self.ou_dislocation_usd: float = 0.08  # Start with a slight +$0.08 spread

        # Try an initial non-blocking live price fetch in background or quick timeout
        self.try_sync_live_prices(timeout_sec=1.8, normalize_to_500=True)

    def _recalibrate_box_multipliers(self) -> None:
        """
        Calibrates the exact number of shares of each constituent per 1 share of QQQ
        in the Top-5 Normalized Box so that at t=0:
            Sum_{i in Top5} (shares_i * anchor_price_i) == anchor_qqq_price
        """
        self.box_share_weights = {}
        for sym, cfg in CONSTITUENTS.items():
            dollar_alloc = self.anchor_qqq_price * cfg.box_norm_weight
            self.box_share_weights[sym] = dollar_alloc / self.anchor_stock_prices[sym]

    def try_sync_live_prices(
        self, timeout_sec: float = 2.0, normalize_to_500: bool = False
    ) -> Dict[str, float]:
        """
        Attempts to fetch live market prices from Yahoo Finance v8 chart endpoint
        for QQQ, NVDA, AAPL, MSFT, AMZN, META.
        If `normalize_to_500` is True, preserves relative live constituent ratios while
        scaling QQQ baseline to $500.00 for intuitive comparison with the $500.00 textbook example.
        """
        symbols = ["QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META"]
        fetched: Dict[str, float] = {}

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }
        for sym in symbols:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=1d"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    meta = payload["chart"]["result"][0]["meta"]
                    price = float(meta.get("regularMarketPrice") or meta.get("previousClose") or 0.0)
                    if price > 1.0:
                        fetched[sym] = round(price, 2)
            except Exception:
                break

        if len(fetched) == len(symbols):
            self.last_live_sync_ts = time.time()
            if normalize_to_500:
                self.anchor_qqq_price = 500.00
                self.data_source_mode = "YAHOO_LIVE_CALIBRATED_500"
            else:
                self.anchor_qqq_price = fetched["QQQ"]
                self.data_source_mode = "YAHOO_LIVE_RAW"

            for sym in CONSTITUENTS:
                self.anchor_stock_prices[sym] = fetched[sym]

            self.stock_prices = dict(self.anchor_stock_prices)
            self.prev_stock_prices = dict(self.anchor_stock_prices)
            self.qqq_price = self.anchor_qqq_price
            self.prev_qqq_price = self.anchor_qqq_price
            self._recalibrate_box_multipliers()
        else:
            # Use clean institutional $500.00 reference anchor
            self.data_source_mode = "SIMULATED_500_BENCHMARK"
            self.anchor_qqq_price = 500.00
            self.stock_prices = dict(self.anchor_stock_prices)
            self.prev_stock_prices = dict(self.anchor_stock_prices)
            self.qqq_price = 500.00
            self.prev_qqq_price = 500.00
            self._recalibrate_box_multipliers()

        return {"QQQ": self.qqq_price, **self.stock_prices}

    def inject_shock(self, shock_type: str) -> Dict[str, object]:
        """
        Injects an immediate market microstructure event to demonstrate Basket vs. Box
        arbitrage and PyTorch Lead-Lag prediction.
        """
        self.shock_Banner_ttl = 45  # Display active shock badge for ~45 ticks

        if shock_type == "JANE_STREET_500_45":
            # Exact user prompt scenario: QQQ = $500.45 while the stocks inside it add up to $500.00!
            self.anchor_qqq_price = 500.00
            for sym, cfg in CONSTITUENTS.items():
                self.stock_prices[sym] = self.anchor_stock_prices[sym]
                self.prev_stock_prices[sym] = self.anchor_stock_prices[sym]
            self._recalibrate_box_multipliers()
            self.ou_dislocation_usd = 0.45
            self.qqq_price = 500.45
            self.prev_qqq_price = 500.08
            self.active_shock_label = "Jane Street Scenario: QQQ $500.45 vs. Box $500.00 (+$0.45 Gap)"

        elif shock_type == "NVDA_BREAKOUT":
            # NVDA jumps +1.85% immediately; QQQ hasn't reacted yet (lagged by 2-3 ticks = 120-180ms)!
            self.prev_stock_prices["NVDA"] = self.stock_prices["NVDA"]
            self.stock_prices["NVDA"] *= 1.0185
            # Queue delayed reaction for QQQ over next 3 ticks
            self.propagation_queue.append(16.5)
            self.propagation_queue.append(12.0)
            # Because NVDA jumped instantly while QQQ hasn't moved yet, Box > QQQ (negative gap)
            self.ou_dislocation_usd = -0.38
            self.active_shock_label = "NVDA +1.85% Lead Breakout (QQQ Lagging by ~120ms)"

        elif shock_type == "MEGA_CAP_DIP":
            # NVDA, AAPL, MSFT drop sharply (-1.4%) while QQQ is still trading high -> +$0.52 Overpriced QQQ
            for sym, drop in [("NVDA", -0.016), ("AAPL", -0.012), ("MSFT", -0.011), ("AMZN", -0.009)]:
                self.prev_stock_prices[sym] = self.stock_prices[sym]
                self.stock_prices[sym] *= 1.0 + drop
            self.propagation_queue.append(-18.0)
            self.propagation_queue.append(-11.0)
            self.ou_dislocation_usd = 0.52
            self.active_shock_label = "Mega-Cap Flash Dip: Box Drops First -> QQQ +$0.52 Overpriced"

        elif shock_type == "BOX_DISCOUNT_SWEEP":
            # Aggressive ETF sell program pushes QQQ -$0.44 below the constituent basket
            self.ou_dislocation_usd = -0.44
            self.active_shock_label = "ETF Sell Sweep: QQQ -$0.44 Underpriced vs. Constituent Box"

        elif shock_type == "RESET_500":
            self.anchor_qqq_price = 500.00
            self.stock_prices = dict(self.anchor_stock_prices)
            self.prev_stock_prices = dict(self.anchor_stock_prices)
            self._recalibrate_box_multipliers()
            self.ou_dislocation_usd = 0.0
            self.qqq_price = 500.00
            self.prev_qqq_price = 500.00
            self.active_shock_label = "Equilibrium Reset ($500.00 Parity)"

        return {
            "status": "ok",
            "shock_type": shock_type,
            "active_shock_label": self.active_shock_label,
        }

    def step_tick(self) -> Dict[str, object]:
        """
        Advances the high-frequency market state by 1 tick (~60ms):
        - Evolves NVDA with occasional Hawkes-like momentum impulses
        - Evolves AAPL, MSFT, AMZN, META with cross-sectional correlation to NVDA
        - Computes exact Synthetic Box Price ($) from constituent prices & multipliers
        - Evolves QQQ with a 2-3 tick lag behind NVDA impulses + Ornstein-Uhlenbeck basis dislocation
        """
        self.tick_count += 1
        if self.shock_Banner_ttl > 0:
            self.shock_Banner_ttl -= 1
            if self.shock_Banner_ttl == 0:
                self.active_shock_label = "Normal Microstructure Flow"

        # 1. Generate common Nasdaq market factor + occasional NVDA momentum burst
        market_factor_bps = float(self.rng.normal(0.0, 0.65))
        nvda_burst_bps = 0.0
        if self.rng.random() < 0.08:
            # Spontaneous microstructure momentum burst in NVDA
            nvda_burst_bps = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(5.5, 13.5))

        returns_bps: Dict[str, float] = {}
        for sym, cfg in CONSTITUENTS.items():
            self.prev_stock_prices[sym] = self.stock_prices[sym]
            idio_bps = float(self.rng.normal(0.0, cfg.volatility_bps * 0.45))
            if sym == "NVDA":
                ret_bps = market_factor_bps + nvda_burst_bps + idio_bps
            else:
                # Sympathetic correlation to market factor + slight NVDA spillover
                ret_bps = 0.85 * market_factor_bps + 0.18 * nvda_burst_bps + idio_bps

            # Gentle anchor mean-reversion so prices stay near realistic levels over hours
            anchor_dev = (self.stock_prices[sym] - self.anchor_stock_prices[sym]) / self.anchor_stock_prices[sym]
            ret_bps -= float(anchor_dev * 18.0)

            self.stock_prices[sym] = max(1.0, self.stock_prices[sym] * (1.0 + ret_bps / 10000.0))
            returns_bps[sym] = ret_bps

        # 2. Compute Synthetic Box Fair Value ($) from live constituent prices
        box_price_usd = sum(
            self.box_share_weights[sym] * self.stock_prices[sym] for sym in CONSTITUENTS
        )
        prev_box_price_usd = sum(
            self.box_share_weights[sym] * self.prev_stock_prices[sym] for sym in CONSTITUENTS
        )
        box_ret_bps = ((box_price_usd - prev_box_price_usd) / max(prev_box_price_usd, 1.0)) * 10000.0

        # 3. Push current NVDA + Basket impulse into the lead-lag propagation queue
        # QQQ only absorbs ~45% of an impulsive constituent move immediately; the rest arrives 2-3 ticks later
        unabsorbed_impulse_bps = 0.55 * (0.45 * returns_bps["NVDA"] + 0.55 * box_ret_bps)
        self.propagation_queue.append(unabsorbed_impulse_bps)
        lagged_impulse_bps = float(self.propagation_queue.popleft()) if self.propagation_queue else 0.0

        # 4. Ornstein-Uhlenbeck basis dislocation (with occasional ETF liquidity sweeps)
        # Mean reversion speed kappa = 0.11 per tick, plus microstructure noise
        ou_noise = float(self.rng.normal(0.0, 0.028))
        if self.rng.random() < 0.045 and abs(self.ou_dislocation_usd) < 0.18:
            # Spontaneous institutional ETF sweep creating a $0.25 - $0.48 arbitrage gap
            ou_noise += float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.24, 0.46))

        # When NVDA jumps first, QQQ initially lags behind the Box
        lead_lag_drag_usd = -box_price_usd * (0.48 * returns_bps["NVDA"] * 0.25 / 10000.0)
        lagged_catchup_usd = box_price_usd * (lagged_impulse_bps * 0.18 / 10000.0)

        self.ou_dislocation_usd = float(
            np.clip(
                0.89 * self.ou_dislocation_usd + ou_noise + lead_lag_drag_usd + lagged_catchup_usd,
                -0.85,
                0.85,
            )
        )

        self.prev_qqq_price = self.qqq_price
        self.qqq_price = box_price_usd + self.ou_dislocation_usd
        qqq_ret_bps = ((self.qqq_price - self.prev_qqq_price) / max(self.prev_qqq_price, 1.0)) * 10000.0

        # 5. Build detailed constituent breakdown
        constituent_rows: List[Dict[str, object]] = []
        for sym, cfg in CONSTITUENTS.items():
            p = self.stock_prices[sym]
            anchor_p = self.anchor_stock_prices[sym]
            half_sp_usd = p * (cfg.half_spread_bps / 10000.0)
            box_contribution_usd = self.box_share_weights[sym] * p
            day_chg_pct = ((p - anchor_p) / anchor_p) * 100.0

            constituent_rows.append(
                {
                    "symbol": sym,
                    "name": cfg.name,
                    "price": round(p, 2),
                    "bid": round(p - half_sp_usd, 2),
                    "ask": round(p + half_sp_usd, 2),
                    "tick_ret_bps": round(returns_bps[sym], 2),
                    "day_chg_pct": round(day_chg_pct, 3),
                    "qqq_weight_pct": round(cfg.qqq_weight * 100.0, 2),
                    "box_norm_weight_pct": round(cfg.box_norm_weight * 100.0, 2),
                    "shares_per_qqq": round(self.box_share_weights[sym], 4),
                    "box_contribution_usd": round(box_contribution_usd, 2),
                }
            )

        gap_usd = self.qqq_price - box_price_usd
        gap_bps = (gap_usd / max(box_price_usd, 1.0)) * 10000.0

        qqq_half_spread = self.qqq_price * (0.45 / 10000.0)

        return {
            "timestamp": time.time(),
            "tick_count": self.tick_count,
            "data_source_mode": self.data_source_mode,
            "active_shock_label": self.active_shock_label,
            "qqq": {
                "symbol": "QQQ",
                "price": round(self.qqq_price, 2),
                "bid": round(self.qqq_price - qqq_half_spread, 2),
                "ask": round(self.qqq_price + qqq_half_spread, 2),
                "tick_ret_bps": round(qqq_ret_bps, 2),
            },
            "box": {
                "price": round(box_price_usd, 2),
                "tick_ret_bps": round(box_ret_bps, 2),
                "top5_raw_qqq_weight_pct": round(
                    sum(c.qqq_weight for c in CONSTITUENTS.values()) * 100.0, 2
                ),
            },
            "spread": {
                "gap_usd": round(gap_usd, 4),
                "gap_bps": round(gap_bps, 2),
            },
            "returns_bps": returns_bps,
            "qqq_ret_bps": qqq_ret_bps,
            "constituents": constituent_rows,
        }
