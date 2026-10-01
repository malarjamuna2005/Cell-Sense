"""
CellSense — API Endpoint Tests.

Uses FastAPI TestClient to validate all REST endpoints.
The model is initialised manually before tests run so that the predict
and telemetry endpoints are functional.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Pre-initialise the engine singleton BEFORE importing the FastAPI app,
# because the lifespan event may not fire consistently in TestClient.
from backend.inference_engine import CellSenseSOHInferenceEngine

MODEL_PATH = os.path.join(ROOT, "cellsense_final_soh_model.joblib")
_engine = CellSenseSOHInferenceEngine(MODEL_PATH)
CellSenseSOHInferenceEngine._instance = _engine

# Now import app (and force-init its module-level data sources)
import backend.main as _main
from backend.data_source import MockDataSource, PythonModelDataSource

_main._engine = _engine
_main._model_data_source = PythonModelDataSource()
_main._demo_data_source = MockDataSource()
_main._app_start_time = __import__("time").time()

from fastapi.testclient import TestClient

client = TestClient(_main.app)

# ── Valid telemetry payload ────────────────────────────────────

VALID_TELEMETRY = {
    "telemetry": {
        "voltage_mean": 3.825,
        "voltage_min": 2.6995,
        "voltage_max": 4.210,
        "voltage_std": 0.275,
        "current_mean": -0.045,
        "current_min": -0.550,
        "current_max": 0.870,
        "current_std": 0.534,
        "discharge_duration_sec": 420000.0,
        "energy_Wh": 9.5,
        "power_mean": -0.05,
        "power_std": 2.055,
        "voltage_slope": -1.2e-6,
        "current_slope": -4.5e-6,
    }
}


# ── Health endpoint ───────────────────────────────────────────

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "uptime_sec" in data
    assert data["model_loaded"] is True


# ── Device status ─────────────────────────────────────────────

def test_device_status():
    r = client.get("/api/device/status")
    assert r.status_code == 200
    data = r.json()
    assert "connected" in data
    assert "demo_mode" in data
    assert data["model_loaded"] is True


# ── Model info ────────────────────────────────────────────────

def test_ml_model_info():
    r = client.get("/api/ml/model")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "READY"
    assert data["model_type"] == "HistGradientBoostingRegressor"
    assert data["input_feature_count"] == 14
    assert len(data["feature_names"]) == 14


# ── Predict endpoint ─────────────────────────────────────────

def test_predict_valid():
    r = client.post("/api/ml/predict", json=VALID_TELEMETRY)
    assert r.status_code == 200
    data = r.json()
    assert "soh" in data
    assert data["soh"] is not None
    assert isinstance(data["soh"], (int, float))
    assert data["inference_status"] == "success"
    assert data["inference_latency_ms"] > 0
    assert data["source"] in ("python_model", "demo")


def test_predict_missing_feature():
    bad = {"telemetry": {"voltage_mean": 3.8}}  # missing 13 features
    r = client.post("/api/ml/predict", json=bad)
    assert r.status_code == 422


def test_predict_empty_body():
    r = client.post("/api/ml/predict", json={})
    assert r.status_code == 422


# ── History endpoint ──────────────────────────────────────────

def test_history():
    # Ensure at least one prediction exists
    client.post("/api/ml/predict", json=VALID_TELEMETRY)
    r = client.get("/api/ml/history")
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert data["count"] >= 1
    # Verify history entries have required fields
    entry = data["predictions"][-1]
    assert "soh" in entry
    assert "timestamp" in entry
    assert "source" in entry


# ── Battery status ────────────────────────────────────────────

def test_battery_status():
    r = client.get("/api/battery/status")
    assert r.status_code == 200
    data = r.json()
    assert "soh" in data
    assert "prediction_count" in data
    assert data["status"] == "ready"


# ── Alerts ────────────────────────────────────────────────────

def test_alerts():
    r = client.get("/api/alerts")
    assert r.status_code == 200
    data = r.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_acknowledge_nonexistent_alert():
    r = client.post("/api/alerts/nonexistent/acknowledge")
    assert r.status_code == 404


# ── Telemetry endpoint ───────────────────────────────────────

def test_telemetry():
    r = client.get("/api/telemetry")
    assert r.status_code == 200
    data = r.json()
    # Without hardware, telemetry may be null
    assert "telemetry" in data or "message" in data


# ── Frontend served ──────────────────────────────────────────

def test_frontend_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "CELLSENSE" in r.text


# ── Demo mode control ────────────────────────────────────────

def test_demo_start_stop():
    r = client.post("/api/demo/start")
    assert r.status_code == 200
    assert r.json()["demo_mode"] is True

    r = client.post("/api/demo/stop")
    assert r.status_code == 200
    assert r.json()["demo_mode"] is False


# ── End-to-end integration: telemetry → model → prediction → UI ──

def test_end_to_end_flow():
    """Full integration: send telemetry → get prediction → check history → verify battery status."""
    # 1. Send valid telemetry
    r = client.post("/api/ml/predict", json=VALID_TELEMETRY)
    assert r.status_code == 200
    pred = r.json()
    assert pred["inference_status"] == "success"
    soh = pred["soh"]
    assert soh is not None

    # 2. Verify it appears in history
    r = client.get("/api/ml/history")
    history = r.json()
    assert history["count"] >= 1
    last = history["predictions"][-1]
    assert last["soh"] == soh

    # 3. Verify battery status reflects latest prediction
    r = client.get("/api/battery/status")
    battery = r.json()
    assert battery["soh"] == soh
    assert battery["prediction_count"] >= 1
