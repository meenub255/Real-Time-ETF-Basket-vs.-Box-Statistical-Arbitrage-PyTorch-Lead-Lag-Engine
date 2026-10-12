# Real-Time ETF "Basket vs. Box" Statistical Arbitrage & PyTorch Lead-Lag Engine

An institutional-style statistical arbitrage scanner and quantitative trading terminal that tracks **Invesco QQQ (`QQQ`)** against the real-time sum of its top constituent stocks (**`NVDA`**, **`AAPL`**, **`MSFT`**, **`AMZN`**, **`META`**), detects millisecond **`NVDA → QQQ` lead-lag reactions** using a live **PyTorch Temporal Cross-Attention Neural Network**, renders an interactive **WebGL 3D Price Gap Surface**, and generates fee-adjusted **Multi-Leg Arbitrage Trade Tickets**.

---

## Core Concept: "Basket vs. Box" ETF Arbitrage

The `QQQ` ETF is a basket holding Nasdaq-100 stocks (`NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`, etc.). By no-arbitrage parity, the market price of `QQQ` should equal the weighted sum of its underlying constituent stocks (the **Synthetic Stock Box**):

$$\text{Price Gap (\$)} = P_{\text{QQQ}}(t) - \sum_{i \in \text{Box}} w_i \cdot P_i(t)$$

During fast market moves or order-flow sweeps:
1. **Conversion Arbitrage (`QQQ > Box`)**: `QQQ` might trade at **`$500.45`** while the constituent stocks inside it sum to **`$500.00`**. Quantitative market makers (e.g., Jane Street, Citadel Securities) instantly **sell short the overpriced `QQQ` ETF** and **buy the cheaper constituent stock basket** to lock in the **`$0.45` spread**.
2. **Millisecond Lead-Lag Alpha (`NVDA → QQQ`)**: Mega-cap leaders like **`NVDA`** often react to order-flow impulses `50ms–200ms` before the `QQQ` ETF fully reprices. Detecting an impulsive jump in `NVDA` before `QQQ` reacts provides a millisecond head start to front-run the ETF convergence.

---

## Key Features

### 1. Live `QQQ` vs. Synthetic Constituent Box Tracker
- **Hybrid Market Engine (`backend/market_engine.py`)**: Syncs live baseline prices from the Yahoo Finance v8 API (`QQQ`, `NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`) with automatic fallback to a calibrated `$500.00` parity benchmark, overlaid with a `60ms` high-frequency Ornstein-Uhlenbeck + Hawkes jump-diffusion microstructure stream.
- **Real-Time Constituent Decomposition Table**: Tracks live prices, BBO bid/ask quotes, tick returns (`bps`), raw `QQQ` weights, normalized Box weights, exact share multipliers per `QQQ` share, and real-time dollar contributions summing to the Synthetic Box NAV.
- **1-Click Market Shock Injectors**:
  - **⚡ Jane Street `$500.45` vs `$500.00` Gap**: Locks in the textbook `+$0.45` conversion arbitrage spread.
  - **🚀 `NVDA` `+1.85%` Lead Breakout**: Fires an immediate Nvidia momentum burst while `QQQ` lags behind.
  - **📉 Mega-Cap Flash Dip** & **🌊 `QQQ` `-$0.44` Discount Sweep**.

### 2. PyTorch Lead-Lag Predictor (`TemporalCrossAttentionNet`)
- **Neural Architecture (`backend/lead_lag_model.py`)**: A custom `torch.nn.Module` combining:
  - **Causal 1D Temporal Convolution** over multi-asset tick returns (`NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`, `QQQ`, and `Basis`).
  - **GRU Sequence Encoder** capturing order-flow momentum across `20` lookback ticks.
  - **Cross-Lag Temporal Attention Head** computing attention weights across historical lags (`50ms` to `500ms`) and constituent lead attribution scores.
- **Online Learning**: Continuously updates weights on live realized ticks via `AdamW` and `HuberLoss`, outputting:
  - Optimal **`NVDA → QQQ` Lead Time ($\tau^*$ in ms)** and the full cross-correlation profile across $\tau \in [-300\text{ms}, +300\text{ms}]$.
  - **Predicted $\Delta\text{QQQ}$ (`bps` and `$`)** over the next `200ms` horizon and signal confidence (`0–100%`).

### 3. Interactive WebGL 3D Price Gap Surface
- Plots the live 3D arbitrage topology (`Z = QQQ − Box ($)`):
  - **Mode A — Live Time × Lead-Lag Horizon ($\tau$)**: Real-time rolling 3D surface showing how dislocations propagate and decay across milliseconds.
  - **Mode B — Parametric `NVDA` Shock (`%`) × `QQQ` Reaction Lag (`ms`)**: Visualizes the structural arbitrage ridge across `-1.5%` to `+1.5%` `NVDA` shocks.

### 4. Live Arbitrage Trade Ticket & Execution Blotter
- Displays the exact multi-leg basket order (`SELL SHORT QQQ / BUY STOCK BASKET` or `BUY QQQ / SELL SHORT STOCK BASKET`) for configurable lot sizes (`100`, `500`, `1,000`, `5,000`, or `50,000` Creation Unit shares).
- **Institutional Fee & Slippage Waterfall**:
  - Gross Box Arbitrage Spread Capture (`$`)
  - PyTorch Lead-Lag Front-Running Alpha Boost (`$`)
  - Bid/Ask Midpoint Peg Slippage (`-0.28 bps`)
  - SEC Section 31 + FINRA TAF + Exchange Fees (`-0.09 bps`)
  - Intraday Stock Borrow & Clearing Cost (`-0.04 bps`)
  - **Net Locked-in Profit After Fees (`$` and `bps`)**
- Includes 1-click **⚡ EXECUTE ARBITRAGE BOX** and an automated **AUTO-ARB BOT** with a live **Executed Arbitrage Blotter & Realized PnL Ledger**.

---

## Project Structure

```text
├── app.py                      # FastAPI + WebSocket server (/ws/stream & REST endpoints)
├── requirements.txt            # Python dependencies (FastAPI, Uvicorn, PyTorch, NumPy, etc.)
├── run.bat                     # 1-click Windows launcher script
├── backend/
│   ├── __init__.py
│   ├── market_engine.py        # Hybrid Yahoo Finance sync + 60ms microstructure simulator
│   ├── lead_lag_model.py       # PyTorch TemporalCrossAttentionNet + online SGD inference
│   └── arb_engine.py           # 3D Price Gap Surface, fee waterfall & execution blotter
└── static/
    ├── index.html              # Institutional quant trading terminal layout
    ├── styles.css              # Dark-mode trading desk styling
    └── app.js                  # WebSocket client, Plotly WebGL 3D surface & UI controller
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
*(Or double-click `run.bat` on Windows).*

### 3. Open the Trading Terminal
Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
