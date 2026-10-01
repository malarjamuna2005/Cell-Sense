"""
CellSense - Deep Diagnostic Part 2
Investigating the CORRECT threshold comparison in sklearn's TreePredictor.
Key question: does sklearn compare binned (uint8) against bin_threshold (uint8)
or raw float against num_threshold (float)?
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
import os

print(f"scikit-learn version: {sklearn.__version__}")

FEATURES = [
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
    "current_mean", "current_min", "current_max", "current_std",
    "discharge_duration_sec", "energy_Wh", "power_mean", "power_std",
    "voltage_slope", "current_slope",
]

TEST_VECTOR = [
    3.82, 2.70, 4.20, 0.25,
    -0.05, -0.55, 0.85, 0.53,
    50000, 10.0, -0.10, 2.05,
    -0.00001, -0.00002,
]

m = joblib.load("cellsense_final_soh_model.joblib")
X_df = pd.DataFrame([TEST_VECTOR], columns=FEATURES)
sklearn_pred = float(m.predict(X_df)[0])
print(f"sklearn prediction: {sklearn_pred}")
print()

# Find sklearn source for TreePredictor
try:
    from sklearn.ensemble._hist_gradient_boosting import predictor as pred_module
    src_file = pred_module.__file__
    print(f"TreePredictor source: {src_file}")
except Exception as e:
    print(f"Could not find source: {e}")

# Find sklearn source for gradient_boosting predict
try:
    from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb_module
    import inspect
    src = inspect.getsource(gb_module.BaseHistGradientBoosting._raw_predict)
    print("=" * 60)
    print("_raw_predict SOURCE:")
    print("=" * 60)
    print(src)
except Exception as e:
    print(f"Could not get _raw_predict source: {e}")

print()
print("=" * 60)
print("CHECKING predict_binned vs predict on TreePredictor")
print("=" * 60)

X_np = X_df.to_numpy(dtype=np.float64)
X_binned = m._bin_mapper.transform(X_np)
x_binned = X_binned[0]
print(f"X_binned: {x_binned}")
print(f"X_binned dtype: {X_binned.dtype}")

p0 = m._predictors[0][0]
nodes = p0.nodes
print(f"\nTree 0 node[0] bin_threshold: {nodes[0]['bin_threshold']} (uint8)")
print(f"Tree 0 node[0] num_threshold: {nodes[0]['num_threshold']} (float)")
print(f"Tree 0 node[0] feature_idx: {nodes[0]['feature_idx']}")
print()

# The key question: when sklearn traverses, does it use bin_threshold or num_threshold?
# Let's check by looking at what predict_binned does
print("=" * 60)
print("CORRECT BINNED TRAVERSAL (bin_threshold comparison)")
print("=" * 60)
x_binned_row = X_binned[0]
y = float(m._baseline_prediction[0, 0])
lr = m.learning_rate

for tree_idx, predictors_at_stage in enumerate(m._predictors):
    nodes = predictors_at_stage[0].nodes
    node_idx = 0
    while True:
        node = nodes[node_idx]
        if node['is_leaf']:
            y += lr * float(node['value'])
            break
        feat = int(node['feature_idx'])
        bin_val = int(x_binned_row[feat])      # uint8 binned value
        bin_thr = int(node['bin_threshold'])   # uint8 bin threshold
        
        missing_go_to_left = bool(node['missing_go_to_left'])
        
        # Check for missing (255 is the "missing" bin in sklearn)
        if bin_val == 255:  # missing bin indicator
            node_idx = int(node['left']) if missing_go_to_left else int(node['right'])
        elif bin_val <= bin_thr:
            node_idx = int(node['left'])
        else:
            node_idx = int(node['right'])

print(f"Binned traversal (bin_threshold, lr*leaf): {y}")
print(f"sklearn reference: {sklearn_pred}")
print(f"Difference: {abs(y - sklearn_pred)}")
print()

# Let's also check if the n_bins used for missing is actually 255 or different
print("=" * 60)
print("BIN MAPPER MISSING VALUE INVESTIGATION")
print("=" * 60)
bm = m._bin_mapper
print(f"n_bins: {getattr(bm, 'n_bins', 'N/A')}")
print(f"missing_values_bin_idx: {getattr(bm, 'missing_values_bin_idx_', 'N/A')}")
print(f"Actual uint8 value for missing: 255 (max of uint8)")

# Check what X_binned looks like for a NaN input
X_nan = X_df.copy().to_numpy(dtype=np.float64)
X_nan[0, 0] = np.nan
X_nan_binned = m._bin_mapper.transform(X_nan)
print(f"Binned value for NaN in feature 0: {X_nan_binned[0, 0]}")
print()

print("=" * 60)
print("STEP 3: What does _raw_predict actually call?")
print("=" * 60)
# Try to trace _raw_predict step by step
raw = m._raw_predict(X_np)
print(f"_raw_predict(X_np) = {raw}")
print()

# The _raw_predict in sklearn calls predict_binned internally after binning
# Let's verify by checking it manually with correct bin_threshold comparison
print("=" * 60)
print("FULL CORRECT ACCUMULATION (bin_threshold uint8 comparison)")
print("=" * 60)
missing_bin = int(m._bin_mapper.missing_values_bin_idx_) if hasattr(m._bin_mapper, 'missing_values_bin_idx_') else 255
print(f"missing bin index: {missing_bin}")

x_binned_row = X_binned[0]
y = float(m._baseline_prediction[0, 0])
lr = m.learning_rate

for tree_idx, predictors_at_stage in enumerate(m._predictors):
    nodes = predictors_at_stage[0].nodes
    node_idx = 0
    while True:
        node = nodes[node_idx]
        if node['is_leaf']:
            y += lr * float(node['value'])
            break
        feat = int(node['feature_idx'])
        bin_val = int(x_binned_row[feat])
        bin_thr = int(node['bin_threshold'])
        missing_go_left = bool(node['missing_go_to_left'])
        
        if bin_val == missing_bin:
            node_idx = int(node['left']) if missing_go_left else int(node['right'])
        elif bin_val <= bin_thr:
            node_idx = int(node['left'])
        else:
            node_idx = int(node['right'])

print(f"Correct binned result: {y}")
print(f"sklearn reference: {sklearn_pred}")
print(f"Difference: {abs(y - sklearn_pred)}")
print()

print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
