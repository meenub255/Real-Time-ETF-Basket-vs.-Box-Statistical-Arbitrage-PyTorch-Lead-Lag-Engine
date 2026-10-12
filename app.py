"""
FastAPI + WebSocket Server for the ETF "Basket vs. Box" Arbitrage Scanner (QQQ vs. Top Stocks).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.arb_engine import ArbitrageEngine
from backend.lead_lag_model import PyTorchLeadLagPredictor
from backend.market_engine import HybridMarketEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="ETF Basket vs. Box Arbitrage Scanner (QQQ vs. Top Constituents)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global real-time quantitative engines
TICK_INTERVAL_MS = 120  # ~8.3 ticks/sec for smooth browser WebGL + Canvas rendering
market_engine = HybridMarketEngine(tick_interval_ms=50)
lead_lag_predictor = PyTorchLeadLagPredictor(tick_interval_ms=50, seq_len=20, horizon_ticks=4)
arb_engine = ArbitrageEngine(default_lot_size=1000)

latest_payload: Dict[str, object] = {}
connected_clients: List[WebSocket] = []
stream_paused: bool = False


class ShockRequest(BaseModel):
    shock_type: str


class ConfigRequest(BaseModel):
    qqq_lot_size: Optional[int] = None
    min_net_profit_threshold_usd: Optional[float] = None
    auto_arb_enabled: Optional[bool] = None
    stream_paused: Optional[bool] = None


class SyncPricesRequest(BaseModel):
    normalize_to_500: bool = False


def compute_single_step() -> Dict[str, object]:
    """Runs one full tick cycle across Market Engine, PyTorch Lead-Lag Net, and Arb Engine."""
    global latest_payload
    mkt = market_engine.step_tick()
    pred = lead_lag_predictor.step(
        returns_bps=mkt["returns_bps"],
        qqq_ret_bps=mkt["qqq_ret_bps"],
        basis_bps=mkt["spread"]["gap_bps"],
        qqq_price=mkt["qqq"]["price"],
    )
    arb = arb_engine.evaluate_tick(mkt, pred)

    latest_payload = {
        "market": mkt,
        "lead_lag": asdict(pred),
        "arbitrage": arb,
        "stream_paused": stream_paused,
    }
    return latest_payload


# Warm up 25 ticks so rolling charts and 3D surface are immediately populated on first page load
for _ in range(25):
    compute_single_step()


@app.on_event("startup")
async def start_background_ticker() -> None:
    async def ticker_loop() -> None:
        while True:
            if not stream_paused:
                payload = compute_single_step()
            else:
                payload = latest_payload

            if connected_clients and payload:
                dead_clients: List[WebSocket] = []
                for ws in list(connected_clients):
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead_clients.append(ws)
                for dc in dead_clients:
                    if dc in connected_clients:
                        connected_clients.remove(dc)

            await asyncio.sleep(TICK_INTERVAL_MS / 1000.0)

    asyncio.create_task(ticker_loop())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/snapshot")
async def get_snapshot() -> JSONResponse:
    if not latest_payload:
        compute_single_step()
    return JSONResponse(latest_payload)


@app.post("/api/shock")
async def trigger_shock(req: ShockRequest) -> JSONResponse:
    res = market_engine.inject_shock(req.shock_type)
    # Immediately compute step so even if paused or mid-tick, state updates right away
    payload = compute_single_step()
    # If the user triggered the exact $500.45 vs $500.00 scenario, pin the exact numbers for clarity
    if req.shock_type == "JANE_STREET_500_45":
        market_engine.ou_dislocation_usd = 0.45
        market_engine.qqq_price = round(float(payload["market"]["box"]["price"]) + 0.45, 2)
        payload["market"]["qqq"]["price"] = market_engine.qqq_price
        payload["market"]["spread"]["gap_usd"] = 0.45
        payload["market"]["spread"]["gap_bps"] = round(
            (0.45 / float(payload["market"]["box"]["price"])) * 10000.0, 2
        )
        # Re-evaluate arbitrage ticket with exact +$0.45 gap
        pred_obj = lead_lag_predictor.step(
            returns_bps=payload["market"]["returns_bps"],
            qqq_ret_bps=payload["market"]["qqq_ret_bps"],
            basis_bps=payload["market"]["spread"]["gap_bps"],
            qqq_price=market_engine.qqq_price,
        )
        payload["lead_lag"] = asdict(pred_obj)
        payload["arbitrage"] = arb_engine.evaluate_tick(payload["market"], pred_obj)
    return JSONResponse({"result": res, "snapshot": payload})


@app.post("/api/execute")
async def execute_arbitrage() -> JSONResponse:
    if not latest_payload:
        compute_single_step()
    ticket = latest_payload["arbitrage"]["ticket"]
    fill = arb_engine.execute_ticket(ticket, trigger_source="MANUAL_TRADER")
    latest_payload["arbitrage"]["blotter_summary"] = arb_engine.get_blotter_summary()
    return JSONResponse(
        {
            "status": "filled",
            "fill": fill,
            "blotter_summary": latest_payload["arbitrage"]["blotter_summary"],
        }
    )


@app.post("/api/config")
async def update_config(req: ConfigRequest) -> JSONResponse:
    global stream_paused
    if req.stream_paused is not None:
        stream_paused = bool(req.stream_paused)
    cfg = arb_engine.set_config(
        qqq_lot_size=req.qqq_lot_size,
        min_net_profit_threshold_usd=req.min_net_profit_threshold_usd,
        auto_arb_enabled=req.auto_arb_enabled,
    )
    if latest_payload:
        latest_payload["stream_paused"] = stream_paused
    return JSONResponse({"status": "ok", "config": cfg, "stream_paused": stream_paused})


@app.post("/api/sync_live")
async def sync_live_prices(req: SyncPricesRequest) -> JSONResponse:
    prices = market_engine.try_sync_live_prices(
        timeout_sec=2.5, normalize_to_500=req.normalize_to_500
    )
    payload = compute_single_step()
    return JSONResponse(
        {
            "status": "ok",
            "mode": market_engine.data_source_mode,
            "prices": prices,
            "snapshot": payload,
        }
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        if latest_payload:
            await websocket.send_json(latest_payload)
        while True:
            # Listen for optional client commands over WS
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
(DOCS_DIR / "assets").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

