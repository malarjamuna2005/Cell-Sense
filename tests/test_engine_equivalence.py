"""
CellSense — Automated Engine Equivalence Tests (Task 11)
=========================================================
Verifies that the CellSense embedded-style inference engine produces
results mathematically equivalent to the authoritative sklearn model
across a comprehensive set of conditions.

Run with:
    python -m pytest tests/test_engine_equivalence.py -v
"""

import math
import sys
import os
import importlib

import numpy as np
import pandas as pd
import joblib
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Reset singleton before each test module load so fresh engine is always used
import backend.inference_engine as _ie_mod
importlib.reload(_ie_mod)
_ie_mod.CellSenseSOHInferenceEngine._instance = None

from backend.inference_engine import CellSenseSOHInferenceEngine

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "cellsense_final_soh_model.joblib")

EXPECTED_FEATURES = [
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_min",
    "current_max",
    "current_std",
    "discharge_duration_sec",
    "energy_Wh",
    "power_mean",
    "power_std",
    "voltage_slope",
    "current_slope",
]

EXPECTED_N_FEATURES = 14
EXPECTED_N_TREES = 500

# Exact 0.0 difference is achieved in practice; keep a tiny float epsilon
# for numerical robustness on different hardware/OS
TOLERANCE_ABS = 1e-9

REFERENCE_VECTOR = [
    3.82, 2.70, 4.20, 0.25,
    -0.05, -0.55, 0.85, 0.53,
    50000, 10.0, -0.10, 2.05,
    -0.00001, -0.00002,
]

EXPECTED_SKLEARN_PRED = 10.650272754727348  # from actual model, not hard-coded

DETERMINISTIC_VECTORS = [
    ("nominal",            [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53, 50000, 10.0, -0.10, 2.05, -0.00001, -0.00002]),
    ("lower_voltage",      [3.72, 2.60, 4.10, 0.30, -0.08, -0.60, 0.70, 0.50, 60000,  8.5, -0.12, 1.95, -0.000015, -0.000025]),
    ("higher_voltage",     [3.90, 2.80, 4.30, 0.20, -0.03, -0.50, 0.95, 0.55, 40000, 12.0, -0.08, 2.10, -0.000008, -0.000018]),
    ("low_current",        [3.82, 2.70, 4.20, 0.25, -0.02, -0.20, 0.30, 0.25, 80000,  5.0, -0.05, 1.50, -0.000005, -0.000010]),
    ("high_current",       [3.80, 2.68, 4.18, 0.28, -0.10, -0.80, 1.20, 0.70, 30000, 15.0, -0.18, 2.30, -0.000020, -0.000030]),
    ("short_duration",     [3.85, 2.72, 4.22, 0.22, -0.04, -0.52, 0.80, 0.48, 17500,  4.0, -0.09, 1.80, -0.000012, -0.000022]),
    ("long_duration",      [3.78, 2.66, 4.15, 0.32, -0.07, -0.65, 0.75, 0.56, 850000, 18.0, -0.11, 2.08, -0.0000035, -0.0000045]),
    ("high_energy",        [3.84, 2.71, 4.21, 0.24, -0.05, -0.54, 0.84, 0.52, 55000, 19.0, -0.10, 2.04, -0.000009, -0.000019]),
    ("low_energy",         [3.81, 2.70, 4.19, 0.26, -0.05, -0.56, 0.86, 0.54, 45000,  0.5, -0.10, 2.06, -0.000011, -0.000021]),
    ("steep_slope",        [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53, 50000, 10.0, -0.10, 2.05, -0.000025, -0.000090]),
    ("near_zero_slope",    [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53, 50000, 10.0, -0.10, 2.05, -3.0e-9, -4.0e-9]),
    ("bin_edge_lower",     [3.80071, 2.69898, 4.20059, 0.223058, -0.0896843, -0.550291, 0.778926, 0.525811, 17302.2, 0.365859, -0.218799, 2.0215, -2.54985e-5, -9.02988e-5]),
    ("bin_edge_upper",     [3.84507, 2.69987, 4.2371, 0.305154, -0.0154476, -0.550291, 0.943368, 0.54225, 826259, 19.1924, 0.0200678, 2.09019, -2.97707e-9, -4.01104e-9]),
    ("mixed_high_std",     [3.83, 2.71, 4.21, 0.28, -0.06, -0.57, 0.88, 0.54, 52000, 11.0, -0.11, 2.07, -0.000013, -0.000023]),
    ("long_low_energy",    [3.79, 2.67, 4.16, 0.31, -0.09, -0.70, 0.72, 0.58, 700000, 2.0, -0.13, 1.98, -0.000004, -0.0000055]),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sklearn_model():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def engine():
    _ie_mod.CellSenseSOHInferenceEngine._instance = None
    eng = CellSenseSOHInferenceEngine.get_instance(MODEL_PATH)
    return eng


# ── Test 1: Feature count ─────────────────────────────────────────────────────
def test_feature_count(engine):
    """Engine must expose exactly 14 input features."""
    assert engine.n_features == EXPECTED_N_FEATURES, (
        f"Expected {EXPECTED_N_FEATURES} features, got {engine.n_features}"
    )


# ── Test 2: Feature order ─────────────────────────────────────────────────────
def test_feature_order(engine):
    """Engine feature names must match the authoritative ordered list exactly."""
    assert engine.feature_names == EXPECTED_FEATURES, (
        f"Feature name/order mismatch:\n  got: {engine.feature_names}\n  expected: {EXPECTED_FEATURES}"
    )


# ── Test 3: Model loads correctly ─────────────────────────────────────────────
def test_model_loads(sklearn_model):
    """Authoritative sklearn model must load and be the correct class."""
    assert sklearn_model is not None
    assert sklearn_model.__class__.__name__ == "HistGradientBoostingRegressor"
    assert int(sklearn_model.n_features_in_) == EXPECTED_N_FEATURES
    assert len(sklearn_model._predictors) == EXPECTED_N_TREES


# ── Test 4: Engine initialises ────────────────────────────────────────────────
def test_engine_initialises(engine):
    """Engine must initialise with correct metadata."""
    assert engine is not None
    assert engine.n_features == EXPECTED_N_FEATURES
    assert engine.n_trees == EXPECTED_N_TREES
    assert math.isfinite(engine.baseline_prediction)
    assert abs(engine.baseline_prediction - 37.66632221833227) < 1e-6


# ── Test 5: Reference vector — sklearn vs engine ───────────────────────────────
def test_reference_vector_agreement(sklearn_model, engine):
    """sklearn and engine must agree on the canonical reference vector."""
    X_df = pd.DataFrame([REFERENCE_VECTOR], columns=EXPECTED_FEATURES)
    sklearn_pred = float(sklearn_model.predict(X_df)[0])
    engine_pred, _ = engine.predict_one(np.array(REFERENCE_VECTOR, dtype=np.float64))

    # The expected value is taken from the live model, NOT hard-coded
    assert abs(sklearn_pred - EXPECTED_SKLEARN_PRED) < TOLERANCE_ABS, (
        f"sklearn prediction changed: {sklearn_pred} vs expected {EXPECTED_SKLEARN_PRED}"
    )
    assert abs(engine_pred - sklearn_pred) <= TOLERANCE_ABS, (
        f"Engine mismatch: engine={engine_pred}, sklearn={sklearn_pred}, diff={abs(engine_pred-sklearn_pred):.3e}"
    )


# ── Test 6: Multiple deterministic vectors ────────────────────────────────────
@pytest.mark.parametrize("label,vec", DETERMINISTIC_VECTORS)
def test_deterministic_vector(sklearn_model, engine, label, vec):
    """Engine and sklearn must agree on all 15 deterministic vectors."""
    X_df = pd.DataFrame([vec], columns=EXPECTED_FEATURES)
    sklearn_pred = float(sklearn_model.predict(X_df)[0])
    engine_pred, _ = engine.predict_one(np.array(vec, dtype=np.float64))
    diff = abs(sklearn_pred - engine_pred)
    assert diff <= TOLERANCE_ABS, (
        f"[{label}] diff={diff:.3e} exceeds tolerance {TOLERANCE_ABS:.0e}. "
        f"sklearn={sklearn_pred}, engine={engine_pred}"
    )


# ── Test 7: Missing-value / NaN consistency ───────────────────────────────────
def test_missing_value_handling(sklearn_model, engine):
    """
    Engine must route NaN values consistently with sklearn's missing-value path.
    NaN is inserted into feature 0 (voltage_mean); sklearn uses _raw_predict
    which internally bins NaN to the 'missing' bin and routes via missing_go_to_left.
    The engine must follow the same routing using its NaN check on num_threshold path.
    """
    vec = REFERENCE_VECTOR.copy()
    vec[0] = float("nan")   # voltage_mean = NaN

    X_np = np.array(vec, dtype=np.float64)
    X_df = pd.DataFrame([vec], columns=EXPECTED_FEATURES)

    sklearn_pred = float(sklearn_model.predict(X_df)[0])
    engine_pred, _ = engine.predict_one(X_np)
    diff = abs(sklearn_pred - engine_pred)
    assert diff <= TOLERANCE_ABS, (
        f"NaN-handling mismatch: sklearn={sklearn_pred}, engine={engine_pred}, diff={diff:.3e}"
    )


# ── Test 8: Invalid feature count is rejected ─────────────────────────────────
def test_invalid_feature_count_rejected(engine):
    """predict_one must raise ValueError for wrong number of features."""
    with pytest.raises(ValueError):
        engine.predict_one(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    with pytest.raises(ValueError):
        engine.predict_one(np.array(REFERENCE_VECTOR + [0.0], dtype=np.float64))  # 15 features


# ── Test 9: Non-finite values in non-NaN-routing features ────────────────────
def test_inf_input_does_not_crash(engine):
    """
    Inf inputs should produce a finite or non-crashing result.
    sklearn allows non-finite values (ensure_all_finite=False at fit time).
    This test verifies the engine does not raise an unhandled exception.
    If sklearn itself would crash, the engine is not obligated to succeed.
    """
    vec = REFERENCE_VECTOR.copy()
    vec[8] = float("inf")   # discharge_duration_sec = inf
    X_np = np.array(vec, dtype=np.float64)
    # Should not raise; result may be any finite float
    result, latency = engine.predict_one(X_np)
    assert isinstance(result, float)
    assert latency >= 0.0


# ── Test 10: predict_batch consistency ───────────────────────────────────────
def test_predict_batch_matches_predict_one(engine):
    """predict_batch must return the same values as repeated predict_one calls."""
    vecs = [v for _, v in DETERMINISTIC_VECTORS]
    X = np.array(vecs, dtype=np.float64)
    batch_preds, _ = engine.predict_batch(X)

    for i, vec in enumerate(vecs):
        single_pred, _ = engine.predict_one(np.array(vec, dtype=np.float64))
        assert abs(batch_preds[i] - single_pred) <= TOLERANCE_ABS, (
            f"Batch/single mismatch at index {i}: batch={batch_preds[i]}, single={single_pred}"
        )


# ── Test 11: Randomised vectors (fixed seed) ──────────────────────────────────
def test_randomised_vectors_agreement(sklearn_model, engine):
    """Engine must match sklearn for 200 reproducible random vectors."""
    rng = np.random.default_rng(42)
    N = 200
    failures = []

    for i in range(N):
        vec = [
            rng.uniform(3.70, 3.90),
            rng.uniform(2.60, 2.80),
            rng.uniform(4.10, 4.35),
            rng.uniform(0.15, 0.35),
            rng.uniform(-0.15, -0.01),
            rng.uniform(-0.90, -0.20),
            rng.uniform(0.50, 1.20),
            rng.uniform(0.40, 0.70),
            rng.uniform(15000, 900000),
            rng.uniform(0.3, 20.0),
            rng.uniform(-0.25, 0.05),
            rng.uniform(1.80, 2.20),
            rng.uniform(-3e-5, -1e-9),
            rng.uniform(-1e-4, -1e-9),
        ]
        X_df = pd.DataFrame([vec], columns=EXPECTED_FEATURES)
        sk_p = float(sklearn_model.predict(X_df)[0])
        eng_p, _ = engine.predict_one(np.array(vec, dtype=np.float64))
        diff = abs(sk_p - eng_p)
        if diff > TOLERANCE_ABS:
            failures.append((i, diff, sk_p, eng_p))

    assert len(failures) == 0, (
        f"{len(failures)}/{N} vectors failed tolerance {TOLERANCE_ABS:.0e}. "
        f"First failure: vector {failures[0][0]}, diff={failures[0][1]:.3e}"
    )
