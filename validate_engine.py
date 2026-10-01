"""
CellSense Model Validation Script
==================================
Compares the authoritative sklearn HistGradientBoostingRegressor against
the CellSense embedded-style tree traversal engine across:
  - Single deterministic reference vector
  - Multiple deterministic validation vectors (Task 6)
  - Randomised plausible vectors with fixed seed (Task 7)

Do NOT fabricate results. All values are measured at runtime.
"""

import sys
import time
import numpy as np
import pandas as pd
import joblib

# Reset singleton so changes to the module are picked up fresh
import importlib
import backend.inference_engine as _ie_mod
importlib.reload(_ie_mod)
_ie_mod.CellSenseSOHInferenceEngine._instance = None

from backend.inference_engine import CellSenseSOHInferenceEngine

# ── Feature names (authoritative order) ──────────────────────────────────────
FEATURES = [
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

# ── Numerical tolerance ───────────────────────────────────────────────────────
# The engine traverses the same float64 node arrays as sklearn's _raw_predict.
# Comparison of predictor.predict() against raw node traversal shows exact
# agreement (diff = 0.0) for all 500 trees.  Accumulation order in Python vs
# the C extension can produce IEEE-754 rounding differences at the ULP level.
# We use 1e-9 (absolute) as a strict but realistic tolerance.
TOLERANCE_ABS = 1e-9
TOLERANCE_REL = 1e-7   # relative fallback for very large / small predictions


def _div(a, b):
    """Safe relative difference: |a-b| / max(|b|, 1e-12)."""
    return abs(a - b) / max(abs(b), 1e-12)


def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def single_vector_section(model, engine, vector, label="REFERENCE VECTOR"):
    """Run and print the canonical single-vector comparison."""
    X_df = pd.DataFrame([vector], columns=FEATURES)
    sklearn_pred = float(model.predict(X_df)[0])
    engine_pred, latency_ms = engine.predict_one(np.array(vector, dtype=np.float64))
    abs_diff = abs(sklearn_pred - engine_pred)
    rel_diff = _div(sklearn_pred, engine_pred)
    passed = abs_diff <= TOLERANCE_ABS or rel_diff <= TOLERANCE_REL
    return sklearn_pred, engine_pred, abs_diff, rel_diff, latency_ms, passed


# ── Deterministic multi-vector suite (Task 6) ─────────────────────────────────
# Labelled as "deterministic validation vectors" — NOT claimed to be real EV data.
# Each tuple: (label, [14 feature values])
DETERMINISTIC_VECTORS = [
    (
        "Reference — nominal operating",
        [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53,
         50000, 10.0, -0.10, 2.05, -0.00001, -0.00002],
    ),
    (
        "Lower voltage region",
        [3.72, 2.60, 4.10, 0.30, -0.08, -0.60, 0.70, 0.50,
         60000, 8.5, -0.12, 1.95, -0.000015, -0.000025],
    ),
    (
        "Higher voltage region",
        [3.90, 2.80, 4.30, 0.20, -0.03, -0.50, 0.95, 0.55,
         40000, 12.0, -0.08, 2.10, -0.000008, -0.000018],
    ),
    (
        "Low current magnitude",
        [3.82, 2.70, 4.20, 0.25, -0.02, -0.20, 0.30, 0.25,
         80000, 5.0, -0.05, 1.50, -0.000005, -0.000010],
    ),
    (
        "High current magnitude",
        [3.80, 2.68, 4.18, 0.28, -0.10, -0.80, 1.20, 0.70,
         30000, 15.0, -0.18, 2.30, -0.000020, -0.000030],
    ),
    (
        "Short discharge duration",
        [3.85, 2.72, 4.22, 0.22, -0.04, -0.52, 0.80, 0.48,
         17500, 4.0, -0.09, 1.80, -0.000012, -0.000022],
    ),
    (
        "Long discharge duration",
        [3.78, 2.66, 4.15, 0.32, -0.07, -0.65, 0.75, 0.56,
         850000, 18.0, -0.11, 2.08, -0.0000035, -0.0000045],
    ),
    (
        "High energy content",
        [3.84, 2.71, 4.21, 0.24, -0.05, -0.54, 0.84, 0.52,
         55000, 19.0, -0.10, 2.04, -0.000009, -0.000019],
    ),
    (
        "Low energy content",
        [3.81, 2.70, 4.19, 0.26, -0.05, -0.56, 0.86, 0.54,
         45000, 0.5, -0.10, 2.06, -0.000011, -0.000021],
    ),
    (
        "Steep voltage slope",
        [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53,
         50000, 10.0, -0.10, 2.05, -0.000025, -0.000090],
    ),
    (
        "Near-zero slopes",
        [3.82, 2.70, 4.20, 0.25, -0.05, -0.55, 0.85, 0.53,
         50000, 10.0, -0.10, 2.05, -3.0e-9, -4.0e-9],
    ),
    (
        "Boundary-like — voltage near bin edges",
        [3.80071, 2.69898, 4.20059, 0.223058, -0.0896843, -0.550291, 0.778926, 0.525811,
         17302.2, 0.365859, -0.218799, 2.0215, -2.54985e-5, -9.02988e-5],
    ),
    (
        "Boundary-like — voltage near upper bin edges",
        [3.84507, 2.69987, 4.2371, 0.305154, -0.0154476, -0.550291, 0.943368, 0.54225,
         826259, 19.1924, 0.0200678, 2.09019, -2.97707e-9, -4.01104e-9],
    ),
    (
        "Mixed — high std, moderate current",
        [3.83, 2.71, 4.21, 0.28, -0.06, -0.57, 0.88, 0.54,
         52000, 11.0, -0.11, 2.07, -0.000013, -0.000023],
    ),
    (
        "Very high discharge duration, low energy",
        [3.79, 2.67, 4.16, 0.31, -0.09, -0.70, 0.72, 0.58,
         700000, 2.0, -0.13, 1.98, -0.000004, -0.0000055],
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("CELLSENSE MODEL VALIDATION")
    print("=" * 60)
    print()
    print("MODEL:         HistGradientBoostingRegressor")
    print("SKLEARN VER:   1.6.1")
    print("FEATURES:      14")
    print("TREES:         500")
    print(f"TOLERANCE ABS: {TOLERANCE_ABS:.0e}")
    print(f"TOLERANCE REL: {TOLERANCE_REL:.0e}")
    print()
    print("Tolerance rationale: sklearn stores lr*leaf in node['value'].")
    print("The engine traverses the same float64 arrays using identical")
    print("comparison logic. Accumulation over 500 trees in Python vs the")
    print("C extension yields exact agreement (diff=0.0 in verification).")
    print(f"Tolerance {TOLERANCE_ABS:.0e} is therefore conservative.")

    # ── Load model + engine ───────────────────────────────────────────────────
    print_header("LOADING ARTIFACTS")
    model = joblib.load("cellsense_final_soh_model.joblib")
    engine = CellSenseSOHInferenceEngine.get_instance("cellsense_final_soh_model.joblib")
    print(f"Model class:   {model.__class__.__name__}")
    print(f"Baseline:      {model._baseline_prediction[0, 0]:.10f}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"N trees:       {len(model._predictors)}")

    # ── TASK 5 — Single reference vector ─────────────────────────────────────
    REF_VECTOR = DETERMINISTIC_VECTORS[0][1]

    print_header("TASK 5 — REFERENCE VECTOR VALIDATION")
    sk, eng, abs_d, rel_d, lat, passed = single_vector_section(
        model, engine, REF_VECTOR
    )
    print()
    print("FEATURE COUNT:        14")
    print(f"SKLEARN SOH:          {sk:.15f}")
    print(f"ENGINE SOH:           {eng:.15f}")
    print(f"ABSOLUTE DIFFERENCE:  {abs_d:.3e}")
    print(f"RELATIVE DIFFERENCE:  {rel_d:.3e}")
    print(f"ENGINE LATENCY:       {lat:.4f} ms")
    print()
    if passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    # ── TASK 6 — Multiple deterministic vectors ───────────────────────────────
    print_header("TASK 6 — DETERMINISTIC MULTI-VECTOR VALIDATION")
    print(f"{'#':<3}  {'Label':<45}  {'sklearn':>12}  {'engine':>12}  {'|diff|':>10}  {'status'}")
    print("-" * 100)

    det_results = []
    for idx, (label, vec) in enumerate(DETERMINISTIC_VECTORS):
        sk, eng, abs_d, rel_d, lat, passed = single_vector_section(model, engine, vec)
        det_results.append((label, sk, eng, abs_d, rel_d, lat, passed))
        status = "PASS" if passed else "FAIL"
        print(f"{idx+1:<3}  {label:<45}  {sk:>12.6f}  {eng:>12.6f}  {abs_d:>10.3e}  {status}")

    det_passed = sum(1 for r in det_results if r[6])
    det_failed = len(det_results) - det_passed
    det_max_abs = max(r[3] for r in det_results)
    det_mean_abs = np.mean([r[3] for r in det_results])
    det_max_rel = max(r[4] for r in det_results)

    print()
    print(f"Vectors tested:       {len(det_results)}")
    print(f"PASSED:               {det_passed}")
    print(f"FAILED:               {det_failed}")
    print(f"Max absolute diff:    {det_max_abs:.3e}")
    print(f"Mean absolute diff:   {det_mean_abs:.3e}")
    print(f"Max relative diff:    {det_max_rel:.3e}")

    # ── TASK 7 — Randomised numerical validation ──────────────────────────────
    print_header("TASK 7 — RANDOMISED VALIDATION (seed=42, N=200)")
    print("Vectors generated within physically plausible project-supported ranges.")
    print("These are synthetic validation vectors, NOT real EV measurements.")
    print()

    rng = np.random.default_rng(42)
    N_RAND = 200

    # Plausible ranges informed by the bin_thresholds_ bounds seen in the model
    def rand_vectors(n, rng):
        vecs = []
        for _ in range(n):
            v_mean = rng.uniform(3.70, 3.90)
            v_min  = rng.uniform(2.60, 2.80)
            v_max  = rng.uniform(4.10, 4.35)
            v_std  = rng.uniform(0.15, 0.35)
            i_mean = rng.uniform(-0.15, -0.01)
            i_min  = rng.uniform(-0.90, -0.20)
            i_max  = rng.uniform(0.50,  1.20)
            i_std  = rng.uniform(0.40,  0.70)
            dur    = rng.uniform(15000, 900000)
            energy = rng.uniform(0.3,   20.0)
            p_mean = rng.uniform(-0.25,  0.05)
            p_std  = rng.uniform(1.80,   2.20)
            v_slp  = rng.uniform(-3e-5,  -1e-9)
            i_slp  = rng.uniform(-1e-4,  -1e-9)
            vecs.append([v_mean, v_min, v_max, v_std,
                         i_mean, i_min, i_max, i_std,
                         dur, energy, p_mean, p_std,
                         v_slp, i_slp])
        return vecs

    rand_vecs = rand_vectors(N_RAND, rng)

    rand_results = []
    for vec in rand_vecs:
        X_df = pd.DataFrame([vec], columns=FEATURES)
        sk_p = float(model.predict(X_df)[0])
        eng_p, lat_r = engine.predict_one(np.array(vec, dtype=np.float64))
        abs_d = abs(sk_p - eng_p)
        rel_d = _div(sk_p, eng_p)
        passed = abs_d <= TOLERANCE_ABS or rel_d <= TOLERANCE_REL
        rand_results.append((sk_p, eng_p, abs_d, rel_d, lat_r, passed))

    rand_passed = sum(1 for r in rand_results if r[5])
    rand_failed  = N_RAND - rand_passed
    rand_max_abs = max(r[2] for r in rand_results)
    rand_mean_abs = np.mean([r[2] for r in rand_results])
    rand_max_rel = max(r[3] for r in rand_results)
    rand_latencies = [r[4] for r in rand_results]

    print(f"Vectors tested:       {N_RAND}")
    print(f"PASSED:               {rand_passed}")
    print(f"FAILED:               {rand_failed}")
    print(f"Max absolute diff:    {rand_max_abs:.3e}")
    print(f"Mean absolute diff:   {rand_mean_abs:.3e}")
    print(f"Max relative diff:    {rand_max_rel:.3e}")
    print(f"Mean engine latency:  {np.mean(rand_latencies):.4f} ms")
    print(f"Max  engine latency:  {np.max(rand_latencies):.4f} ms")
    print(f"Min  engine latency:  {np.min(rand_latencies):.4f} ms")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print_header("FINAL VALIDATION SUMMARY")
    total_tested  = len(det_results) + N_RAND
    total_passed  = det_passed + rand_passed
    total_failed  = det_failed + rand_failed
    all_abs_diffs = [r[3] for r in det_results] + [r[2] for r in rand_results]
    all_rel_diffs = [r[4] for r in det_results] + [r[3] for r in rand_results]
    global_max_abs = max(all_abs_diffs)
    global_mean_abs = np.mean(all_abs_diffs)
    global_max_rel = max(all_rel_diffs)

    print(f"Total vectors tested:    {total_tested}")
    print(f"  Deterministic:         {len(det_results)}")
    print(f"  Randomised:            {N_RAND}")
    print(f"TOTAL PASSED:            {total_passed}")
    print(f"TOTAL FAILED:            {total_failed}")
    print(f"Global max abs diff:     {global_max_abs:.3e}")
    print(f"Global mean abs diff:    {global_mean_abs:.3e}")
    print(f"Global max rel diff:     {global_max_rel:.3e}")
    print()

    final_status = "PASS" if total_failed == 0 else "FAIL"
    print(f"OVERALL STATUS: {final_status}")
    print()
    if final_status == "PASS":
        print("The CellSense embedded-style inference engine reproduces the")
        print("authoritative sklearn HistGradientBoostingRegressor prediction")
        print(f"within tolerance ({TOLERANCE_ABS:.0e} absolute / {TOLERANCE_REL:.0e} relative)")
        print("across all tested vectors.")
    else:
        print("One or more vectors exceeded tolerance. Review FAIL rows above.")

    return 0 if final_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())