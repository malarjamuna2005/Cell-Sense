"""
CellSense — FastAPI Backend Application.

Serves the ML inference API, live telemetry WebSocket, and the static
frontend dashboard.  The model is loaded ONCE at startup via the singleton
CellSenseSOHInferenceEngine.
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Ensure the project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_source import (
    MockDataSource,
    PythonModelDataSource,
    SOHPrediction,
    TelemetrySnapshot,
)
from backend.feature_builder import (
    REQUIRED_FEATURE_ORDER,
    FeatureValidationError,
    build_features_from_dict,
)
from backend.inference_engine import CellSenseSOHInferenceEngine

# ── Application state (module-level singletons) ─────────────────────────────

_engine: Optional[CellSenseSOHInferenceEngine] = None
_model_data_source: Optional[PythonModelDataSource] = None
_demo_data_source: Optional[MockDataSource] = None
_prediction_history: List[Dict[str, Any]] = []
_alerts: List[Dict[str, Any]] = []
_connected_ws: List[WebSocket] = []
_demo_mode: bool = False
_demo_task: Optional[asyncio.Task] = None
_app_start_time: float = 0.0


# ── Pydantic models ─────────────────────────────────────────────────────────

class TelemetryInput(BaseModel):
    telemetry: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature name → numeric value for all 14 model features",
    )

class PredictResponse(BaseModel):
    soh: float
    timestamp: float
    source: str
    model_version: str
    inference_latency_ms: float
    inference_status: str
    error_message: Optional[str] = None
    features_used: Optional[Dict[str, float]] = None


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_data_source, _demo_data_source, _app_start_time
    _app_start_time = time.time()

    # Locate model artifact
    model_path = os.path.join(PROJECT_ROOT, "cellsense_final_soh_model.joblib")
    if not os.path.exists(model_path):
        print(f"[CellSense] WARNING: model artifact not found at {model_path}")
    else:
        try:
            _engine = CellSenseSOHInferenceEngine(model_path)
            CellSenseSOHInferenceEngine._instance = _engine
            _model_data_source = PythonModelDataSource()
            _demo_data_source = MockDataSource()
            print(f"[CellSense] Model loaded: {_engine.model_class} with {_engine.n_features} features, {_engine.n_trees} trees")
        except Exception as exc:
            print(f"[CellSense] ERROR loading model: {exc}")

    yield

    # Cleanup
    global _demo_task
    if _demo_task and not _demo_task.done():
        _demo_task.cancel()


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CellSense — EV Battery Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(v: float) -> Any:
    """JSON-safe float: replace NaN / Inf with None."""
    if v is None or math.isnan(v) or math.isinf(v):
        return None
    return round(v, 6)

def _add_prediction(pred: SOHPrediction) -> Dict[str, Any]:
    rec = {
        "id": str(uuid.uuid4()),
        "soh": _safe_float(pred.soh_percent),
        "timestamp": pred.timestamp,
        "source": pred.source,
        "inference_latency_ms": round(pred.inference_latency_ms, 3),
        "model_version": pred.model_version,
        "status": pred.status,
        "error_message": pred.error_message,
    }
    _prediction_history.append(rec)
    # Keep bounded
    if len(_prediction_history) > 500:
        _prediction_history.pop(0)
    return rec


async def _broadcast_ws(message: Dict[str, Any]):
    dead = []
    for ws in _connected_ws:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connected_ws.remove(ws)


# ── Static frontend ─────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse("<h1>CellSense</h1><p>Frontend not found. Place index.html in frontend/</p>")


# ── Health & device ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - _app_start_time, 1),
        "model_loaded": _engine is not None,
        "demo_mode": _demo_mode,
    }

@app.get("/api/device/status")
async def device_status():
    src = _demo_data_source if _demo_mode else _model_data_source
    info = src.get_device_info() if src else {}
    return {
        "connected": src.is_connected() if src else False,
        "demo_mode": _demo_mode,
        "model_loaded": _engine is not None,
        "hardware_available": False,
        **info,
    }


# ── Telemetry ────────────────────────────────────────────────────────────────

@app.get("/api/telemetry")
async def get_telemetry():
    src = _demo_data_source if _demo_mode else _model_data_source
    if src is None:
        raise HTTPException(503, "Data source not available")
    snap = src.get_latest_telemetry()
    if snap is None:
        return {"telemetry": None, "message": "No telemetry available. Connect a device or enable demo mode."}
    return {
        "telemetry": snap.features,
        "timestamp": snap.timestamp,
        "source": snap.source,
        "is_simulated": snap.is_simulated,
    }


# ── Battery status ───────────────────────────────────────────────────────────

@app.get("/api/battery/status")
async def battery_status():
    latest = _prediction_history[-1] if _prediction_history else None
    return {
        "soh": latest["soh"] if latest else None,
        "last_prediction_timestamp": latest["timestamp"] if latest else None,
        "prediction_source": latest["source"] if latest else None,
        "prediction_count": len(_prediction_history),
        "demo_mode": _demo_mode,
        "status": "ready" if _engine else "model_unavailable",
    }


# ── ML model info ───────────────────────────────────────────────────────────

@app.get("/api/ml/model")
async def ml_model():
    if _engine is None:
        return {"status": "unavailable", "message": "SOH model could not be loaded"}
    return _engine.get_model_info()

# ── ML predict ───────────────────────────────────────────────────────────────

@app.post("/api/ml/predict", response_model=PredictResponse)
async def ml_predict(body: TelemetryInput):
    if _engine is None:
        raise HTTPException(503, "SOH model is not loaded")

    src = _demo_data_source if _demo_mode else _model_data_source
    if src is None:
        raise HTTPException(503, "Data source not available")

    try:
        pred = src.predict(body.telemetry)
    except FeatureValidationError as exc:
        raise HTTPException(422, detail={"validation_errors": exc.errors})

    rec = _add_prediction(pred)

    # Broadcast to WebSocket clients
    await _broadcast_ws({"type": "prediction", "data": rec})

    return PredictResponse(
        soh=_safe_float(pred.soh_percent),
        timestamp=pred.timestamp,
        source=pred.source,
        model_version=pred.model_version,
        inference_latency_ms=round(pred.inference_latency_ms, 3),
        inference_status=pred.status,
        error_message=pred.error_message,
        features_used=pred.raw_features,
    )


# ── Prediction history ───────────────────────────────────────────────────────

@app.get("/api/ml/history")
async def ml_history():
    return {"predictions": _prediction_history, "count": len(_prediction_history)}


# ── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts():
    return {"alerts": _alerts, "count": len(_alerts)}

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    for a in _alerts:
        if a.get("id") == alert_id:
            a["acknowledged"] = True
            return {"status": "acknowledged", "id": alert_id}
    raise HTTPException(404, "Alert not found")


# ── Demo mode control ────────────────────────────────────────────────────────

@app.post("/api/demo/start")
async def start_demo():
    global _demo_mode, _demo_task
    _demo_mode = True
    if _demo_task is None or _demo_task.done():
        _demo_task = asyncio.create_task(_demo_loop())
    return {"demo_mode": True}

@app.post("/api/demo/stop")
async def stop_demo():
    global _demo_mode, _demo_task
    _demo_mode = False
    if _demo_task and not _demo_task.done():
        _demo_task.cancel()
        _demo_task = None
    return {"demo_mode": False}

async def _demo_loop():
    """Background task that periodically generates demo predictions."""
    while _demo_mode:
        try:
            if _demo_data_source and _engine:
                snap = _demo_data_source.get_latest_telemetry()
                if snap:
                    pred = _demo_data_source.predict(snap.features)
                    rec = _add_prediction(pred)
                    await _broadcast_ws({
                        "type": "telemetry",
                        "data": {
                            "telemetry": snap.features,
                            "timestamp": snap.timestamp,
                            "source": "simulated",
                            "is_simulated": True,
                            "prediction": rec,
                        },
                    })
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"[CellSense Demo] error: {exc}")
        await asyncio.sleep(3.0)


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await ws.accept()
    _connected_ws.append(ws)
    try:
        # Send initial state
        await ws.send_json({
            "type": "init",
            "data": {
                "model_loaded": _engine is not None,
                "demo_mode": _demo_mode,
                "prediction_count": len(_prediction_history),
                "last_prediction": _prediction_history[-1] if _prediction_history else None,
            },
        })
        # Keep alive — wait for messages from client
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "start_demo":
                await start_demo()
                await ws.send_json({"type": "demo_started"})

            elif msg_type == "stop_demo":
                await stop_demo()
                await ws.send_json({"type": "demo_stopped"})

            elif msg_type == "predict":
                telemetry = data.get("telemetry", {})
                if _engine is None:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue
                src = _demo_data_source if _demo_mode else _model_data_source
                try:
                    pred = src.predict(telemetry)
                    rec = _add_prediction(pred)
                    await _broadcast_ws({"type": "prediction", "data": rec})
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _connected_ws:
            _connected_ws.remove(ws)


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
