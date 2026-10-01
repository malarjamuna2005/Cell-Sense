"""
CellSense — Model & Inference Tests.

Tests the actual cellsense_final_soh_model.joblib artifact:
loading, feature compatibility, prediction, and error handling.
"""
import math
import os
import sys
import numpy as np
import pytest

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.inference_engine import CellSenseSOHInferenceEngine
from backend.feature_builder import (
    REQUIRED_FEATURE_ORDER,
    FeatureValidationError,
    build_features_from_dict,
)

MODEL_PATH = os.path.join(ROOT, "cellsense_final_soh_model.joblib")


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Load model once for all tests in this module."""
    return CellSenseSOHInferenceEngine(MODEL_PATH)


@pytest.fixture
def valid_features():
    """Realistic feature dict within the model's trained range."""
    return {
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


# ── 1. Model loads ────────────────────────────────────────────

def test_model_loads(engine):
    assert engine is not None
    assert engine.model_class == "HistGradientBoostingRegressor"


def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"


def test_model_artifact_size():
    size = os.path.getsize(MODEL_PATH)
    assert size > 100_000, "Model artifact suspiciously small"
    assert size < 10_000_000, "Model artifact suspiciously large"


# ── 2. Correct feature count ─────────────────────────────────

def test_feature_count(engine):
    assert engine.n_features == 14


# ── 3. Correct feature order ─────────────────────────────────

def test_feature_names(engine):
    assert engine.feature_names == REQUIRED_FEATURE_ORDER


def test_feature_names_order_matches():
    expected = [
        "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
        "current_mean", "current_min", "current_max", "current_std",
        "discharge_duration_sec", "energy_Wh", "power_mean", "power_std",
        "voltage_slope", "current_slope",
    ]
    assert REQUIRED_FEATURE_ORDER == expected


# ── 4. Preprocessing works ───────────────────────────────────

def test_build_features_returns_array(valid_features):
    arr, report = build_features_from_dict(valid_features)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (14,)
    assert report["is_valid"] is True


# ── 5. Prediction returns numeric SOH ─────────────────────────

def test_prediction_returns_float(engine, valid_features):
    arr, _ = build_features_from_dict(valid_features)
    soh, latency = engine.predict_one(arr)
    assert isinstance(soh, float)
    assert not math.isnan(soh)
    assert not math.isinf(soh)


def test_prediction_latency_measured(engine, valid_features):
    arr, _ = build_features_from_dict(valid_features)
    _, latency = engine.predict_one(arr)
    assert latency > 0, "Inference latency must be positive"


def test_batch_prediction(engine, valid_features):
    arr, _ = build_features_from_dict(valid_features)
    X = np.tile(arr, (5, 1))
    preds, latency = engine.predict_batch(X)
    assert preds.shape == (5,)
    assert latency > 0


# ── 6. Invalid input fails safely ─────────────────────────────

def test_missing_feature_raises(valid_features):
    del valid_features["voltage_mean"]
    with pytest.raises(FeatureValidationError) as exc_info:
        build_features_from_dict(valid_features)
    assert "voltage_mean" in str(exc_info.value)


def test_extra_feature_ignored():
    """Extra keys should NOT cause errors — build only uses required ones."""
    features = {name: 1.0 for name in REQUIRED_FEATURE_ORDER}
    features["bonus_extra_field"] = 999
    arr, _ = build_features_from_dict(features)
    assert arr.shape == (14,)


# ── 7. NaN fails safely ──────────────────────────────────────

def test_nan_feature_raises(valid_features):
    valid_features["voltage_mean"] = float("nan")
    with pytest.raises(FeatureValidationError):
        build_features_from_dict(valid_features)


def test_inf_feature_raises(valid_features):
    valid_features["current_max"] = float("inf")
    with pytest.raises(FeatureValidationError):
        build_features_from_dict(valid_features)


# ── 8. Wrong feature count fails ─────────────────────────────

def test_wrong_shape_raises(engine):
    bad_vec = np.zeros(10)
    with pytest.raises(ValueError):
        engine.predict_one(bad_vec)


# ── 9. Model info is accurate ────────────────────────────────

def test_model_info_not_fabricated(engine):
    info = engine.get_model_info()
    assert info["status"] == "READY"
    assert info["input_feature_count"] == 14
    assert info["model_type"] == "HistGradientBoostingRegressor"
    assert "scikit-learn" in info["framework"]
    assert info["artifact_size_bytes"] == os.path.getsize(MODEL_PATH)
    # Must NOT contain fabricated accuracy/confidence
    for key in ("accuracy", "confidence", "precision", "recall", "r2", "rmse", "mae"):
        assert key not in info, f"Model info must not fabricate '{key}'"


def test_tree_count(engine):
    assert engine.n_trees == 500
    assert engine.learning_rate == 0.05
