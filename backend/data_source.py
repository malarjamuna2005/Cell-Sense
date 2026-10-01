"""
CellSense Data Source Abstraction Layer.

Provides a common interface so the UI can receive telemetry and SOH predictions
from any backend — Python model, MCU over BLE/Wi-Fi, or demo simulation —
without rewriting the frontend.
"""
from __future__ import annotations

import abc
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from backend.feature_builder import REQUIRED_FEATURE_ORDER, build_features_from_dict


# ── Shared data contracts ────────────────────────────────────────────────────

@dataclass
class TelemetrySnapshot:
    """Raw or aggregated telemetry from a single discharge cycle."""
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"          # "measured", "simulated", "mcu"
    is_simulated: bool = False

@dataclass
class SOHPrediction:
    """Result of a single SOH prediction."""
    soh_percent: float
    timestamp: float
    source: str                      # "python_model", "cellsense_mcu", "demo"
    inference_latency_ms: float
    model_version: str
    status: str = "success"          # "success" | "error"
    error_message: Optional[str] = None
    raw_features: Optional[Dict[str, float]] = None


# ── Abstract base ─────────────────────────────────────────────────────────────

class DataSource(abc.ABC):
    """
    Abstract interface consumed by the API layer.

    Current:   PythonModelDataSource → UI
    Future:    MCUDataSource → BLE/Wi-Fi → UI

    The UI never needs to know which implementation is active.
    """

    @abc.abstractmethod
    def get_latest_telemetry(self) -> Optional[TelemetrySnapshot]:
        ...

    @abc.abstractmethod
    def predict(self, telemetry: Dict[str, float]) -> SOHPrediction:
        ...

    @abc.abstractmethod
    def is_connected(self) -> bool:
        ...

    @abc.abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        ...


# ── Demo / Mock data source ──────────────────────────────────────────────────

# Realistic ranges derived from the ACTUAL model's bin_thresholds
_DEMO_RANGES: Dict[str, tuple] = {
    "voltage_mean":           (3.801, 3.845),
    "voltage_min":            (2.699, 2.700),
    "voltage_max":            (4.200, 4.237),
    "voltage_std":            (0.223, 0.305),
    "current_mean":           (-0.090, -0.015),
    "current_min":            (-0.551, -0.550),
    "current_max":            (0.779, 0.943),
    "current_std":            (0.526, 0.542),
    "discharge_duration_sec": (17_302.0, 826_259.0),
    "energy_Wh":              (0.366, 19.192),
    "power_mean":             (-0.219, 0.020),
    "power_std":              (2.022, 2.090),
    "voltage_slope":          (-2.55e-5, -3.0e-9),
    "current_slope":          (-9.03e-5, -4.0e-9),
}


class MockDataSource(DataSource):
    """
    Generates realistic-looking telemetry within the model's trained feature
    ranges.  Every value produced is clearly marked **simulated**.
    """

    def __init__(self, cycle_interval_sec: float = 5.0):
        self._cycle_interval = cycle_interval_sec
        self._cycle = 0
        self._last_snapshot: Optional[TelemetrySnapshot] = None

    # ── helpers ──

    @staticmethod
    def _rand_in(lo: float, hi: float) -> float:
        return lo + random.random() * (hi - lo)

    def _generate_snapshot(self) -> TelemetrySnapshot:
        self._cycle += 1
        features: Dict[str, float] = {}
        for name in REQUIRED_FEATURE_ORDER:
            lo, hi = _DEMO_RANGES[name]
            features[name] = self._rand_in(lo, hi)
        snap = TelemetrySnapshot(
            features=features,
            timestamp=time.time(),
            source="simulated",
            is_simulated=True,
        )
        self._last_snapshot = snap
        return snap

    # ── interface ──

    def get_latest_telemetry(self) -> Optional[TelemetrySnapshot]:
        return self._generate_snapshot()

    def predict(self, telemetry: Dict[str, float]) -> SOHPrediction:
        """Demo prediction — wraps PythonModelDataSource but labels output as demo."""
        from backend.inference_engine import CellSenseSOHInferenceEngine
        engine = CellSenseSOHInferenceEngine.get_instance()
        feature_vec, _ = build_features_from_dict(telemetry)
        soh, latency = engine.predict_one(feature_vec)
        return SOHPrediction(
            soh_percent=round(soh, 4),
            timestamp=time.time(),
            source="demo",
            inference_latency_ms=round(latency, 3),
            model_version=engine.framework,
            raw_features=telemetry,
        )

    def is_connected(self) -> bool:
        return True  # demo is always "connected"

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": "Demo (Simulated)",
            "can_status": "simulated",
            "ble_status": "simulated",
            "firmware": "N/A",
            "connection_type": "demo",
        }


# ── Python model data source ─────────────────────────────────────────────────

class PythonModelDataSource(DataSource):
    """
    Real inference through the ACTUAL CellSense HistGradientBoostingRegressor
    loaded from `cellsense_final_soh_model.joblib`.
    """

    def __init__(self):
        from backend.inference_engine import CellSenseSOHInferenceEngine
        self._engine = CellSenseSOHInferenceEngine.get_instance()

    def get_latest_telemetry(self) -> Optional[TelemetrySnapshot]:
        # Real telemetry would come from CAN/OBD-II — currently none available
        return None

    def predict(self, telemetry: Dict[str, float]) -> SOHPrediction:
        feature_vec, _ = build_features_from_dict(telemetry)
        try:
            soh, latency = self._engine.predict_one(feature_vec)
            return SOHPrediction(
                soh_percent=round(soh, 4),
                timestamp=time.time(),
                source="python_model",
                inference_latency_ms=round(latency, 3),
                model_version=self._engine.framework,
                raw_features=telemetry,
            )
        except Exception as exc:
            return SOHPrediction(
                soh_percent=float("nan"),
                timestamp=time.time(),
                source="python_model",
                inference_latency_ms=0.0,
                model_version=self._engine.framework,
                status="error",
                error_message=str(exc),
            )

    def is_connected(self) -> bool:
        return True  # Python model is always available

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": "Python Inference (Development)",
            "can_status": "not_connected",
            "ble_status": "not_connected",
            "firmware": "N/A",
            "connection_type": "python_local",
        }


# ── Future: MCU data source stub ─────────────────────────────────────────────

class MCUDataSource(DataSource):
    """
    Placeholder for future CellSense MCU hardware integration.
    Will receive telemetry and SOH predictions over BLE / Wi-Fi.
    NOT YET IMPLEMENTED — do NOT claim MCU compatibility.
    """

    def get_latest_telemetry(self) -> Optional[TelemetrySnapshot]:
        return None

    def predict(self, telemetry: Dict[str, float]) -> SOHPrediction:
        return SOHPrediction(
            soh_percent=float("nan"),
            timestamp=time.time(),
            source="cellsense_mcu",
            inference_latency_ms=0.0,
            model_version="Not available",
            status="error",
            error_message="MCU data source is not yet implemented",
        )

    def is_connected(self) -> bool:
        return False

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": "CellSense MCU (Not Connected)",
            "can_status": "not_connected",
            "ble_status": "not_connected",
            "firmware": "Not available",
            "connection_type": "mcu_ble",
        }
