"""
CellSense - Deep Diagnostic Part 4
Key finding: sklearn uses predictor.predict(X_raw, ...) NOT predict_binned.
Verifying that predictor.predict with raw floats + num_threshold is the correct path.
Also investigating what the per-tree contributions are without lr applied.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from functools import partial

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
X_np = X_df.to_numpy(dtype=np.float64)
sklearn_pred = float(m.predict(X_df)[0])
print(f"sklearn prediction: {sklearn_pred}")
print()

print("=" * 60)
print("CALLING predictor.predict CORRECTLY (raw float, no lr)")
print("=" * 60)
# _predict_iterations at inference time calls:
#   predictor.predict(X, known_cat_bitsets, f_idx_map, n_threads)
# and accumulates INTO raw_predictions which starts at baseline
# The key: does predictor.predict return lr*leaf or just leaf?

known_cat_bitsets, f_idx_map = m._bin_mapper.make_known_categories_bitsets()
print(f"known_cat_bitsets shape: {known_cat_bitsets.shape}")
print(f"f_idx_map: {f_idx_map}")
print()

# Call the first predictor's predict
p0 = m._predictors[0][0]
pred0_result = p0.predict(X_np, known_cat_bitsets=known_cat_bitsets, f_idx_map=f_idx_map, n_threads=1)
print(f"p0.predict result: {pred0_result}")
print(f"type: {type(pred0_result)}, dtype: {pred0_result.dtype if hasattr(pred0_result, 'dtype') else 'N/A'}")
print()

# Also check first few predictors
print("First 5 predictor.predict results:")
for i in range(5):
    pi = m._predictors[i][0]
    res = pi.predict(X_np, known_cat_bitsets=known_cat_bitsets, f_idx_map=f_idx_map, n_threads=1)
    print(f"  predictor[{i}].predict: {res[0]}")
print()

# Now manually accumulate with and without lr to see which matches
print("=" * 60)
print("MANUAL ACCUMULATION: baseline + sum(predictor.predict)")
print("(NOT applying lr - testing if predictor.predict already includes lr)")
print("=" * 60)
y_no_lr = float(m._baseline_prediction[0, 0])
y_with_lr = float(m._baseline_prediction[0, 0])
lr = m.learning_rate

all_preds = []
for i, predictors_at_stage in enumerate(m._predictors):
    pi = predictors_at_stage[0]
    pred_val = float(pi.predict(X_np, known_cat_bitsets=known_cat_bitsets, f_idx_map=f_idx_map, n_threads=1)[0])
    all_preds.append(pred_val)
    y_no_lr += pred_val
    y_with_lr += lr * pred_val

print(f"baseline: {m._baseline_prediction[0, 0]}")
print(f"sum of all predictor.predict values: {sum(all_preds)}")
print(f"accumulation WITHOUT extra lr: {y_no_lr}")
print(f"accumulation WITH extra lr: {y_with_lr}")
print(f"sklearn reference: {sklearn_pred}")
print(f"Difference (no extra lr): {abs(y_no_lr - sklearn_pred)}")
print(f"Difference (with extra lr): {abs(y_with_lr - sklearn_pred)}")
print()

# What is the leaf value in tree 0 for our input?
print("=" * 60)
print("TREE 0: Manual traversal using num_threshold (raw float)")
print("=" * 60)
x_raw = np.array(TEST_VECTOR, dtype=np.float64)
nodes = m._predictors[0][0].nodes
node_idx = 0
path = []
while True:
    node = nodes[node_idx]
    path.append(node_idx)
    if node['is_leaf']:
        print(f"Leaf at node {node_idx}: value={node['value']}")
        break
    feat = int(node['feature_idx'])
    val = x_raw[feat]
    threshold = float(node['num_threshold'])
    is_nan = np.isnan(val)
    go_left = node['missing_go_to_left'] if is_nan else (val <= threshold)
    direction = 'left' if go_left else 'right'
    print(f"  node[{node_idx}]: feat[{feat}]({FEATURES[feat]})={val:.6g} vs threshold={threshold:.6g} -> {direction}")
    node_idx = int(node['left']) if go_left else int(node['right'])

print(f"Path: {path}")
print(f"predictor[0].predict = {all_preds[0]}")
print()

# Raw traversal leaf values vs predictor.predict values - are they the same?
print("=" * 60)
print("COMPARING: raw traversal leaf value vs predictor.predict per tree")
print("=" * 60)
mismatches = 0
for i in range(min(10, len(m._predictors))):
    nodes = m._predictors[i][0].nodes
    node_idx = 0
    while True:
        node = nodes[node_idx]
        if node['is_leaf']:
            raw_leaf = float(node['value'])
            break
        feat = int(node['feature_idx'])
        val = x_raw[feat]
        threshold = float(node['num_threshold'])
        is_nan = np.isnan(val)
        go_left = node['missing_go_to_left'] if is_nan else (val <= threshold)
        node_idx = int(node['left']) if go_left else int(node['right'])
    
    predict_val = all_preds[i]
    diff = abs(raw_leaf - predict_val)
    match = "OK" if diff < 1e-12 else "MISMATCH"
    print(f"  tree[{i:3d}]: raw_leaf={raw_leaf:.8f}, predict={predict_val:.8f}, diff={diff:.2e} [{match}]")
    if diff >= 1e-12:
        mismatches += 1

print()
if mismatches == 0:
    print("FINDING: predictor.predict returns the same values as raw traversal leaf nodes")
    print("=> The lr is already embedded in the leaf values!")
else:
    print(f"FINDING: {mismatches} mismatches - traversal path or leaf semantics differ")

print()
print("=" * 60)
print("TESTING: accumulate baseline + sum(leaf_values) WITHOUT lr")
print("=" * 60)
y_raw = float(m._baseline_prediction[0, 0])
for predictors_at_stage in m._predictors:
    nodes = predictors_at_stage[0].nodes
    node_idx = 0
    while True:
        node = nodes[node_idx]
        if node['is_leaf']:
            y_raw += float(node['value'])
            break
        feat = int(node['feature_idx'])
        val = x_raw[feat]
        threshold = float(node['num_threshold'])
        is_nan = np.isnan(val)
        go_left = node['missing_go_to_left'] if is_nan else (val <= threshold)
        node_idx = int(node['left']) if go_left else int(node['right'])

print(f"Result (baseline + sum leaf values, NO lr): {y_raw}")
print(f"sklearn reference: {sklearn_pred}")
print(f"Difference: {abs(y_raw - sklearn_pred)}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
