"""
CellSense - Deep Model Diagnostic Script
Investigates actual sklearn 1.6.1 HistGradientBoostingRegressor internals.
DO NOT MODIFY the model artifact. Read only.
"""
import joblib
import numpy as np
import pandas as pd
import inspect

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

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)
m = joblib.load("cellsense_final_soh_model.joblib")
print(f"Type: {type(m)}")
print(f"Class: {m.__class__.__name__}")
print()

print("=" * 60)
print("SKLEARN REFERENCE PREDICTION")
print("=" * 60)
X_df = pd.DataFrame([TEST_VECTOR], columns=FEATURES)
sklearn_pred = float(m.predict(X_df)[0])
print(f"sklearn prediction: {sklearn_pred}")
print()

print("=" * 60)
print("BASELINE PREDICTION")
print("=" * 60)
bp = m._baseline_prediction
print(f"_baseline_prediction type: {type(bp)}")
print(f"_baseline_prediction dtype: {bp.dtype}")
print(f"_baseline_prediction shape: {bp.shape}")
print(f"_baseline_prediction value: {bp}")
print(f"_baseline_prediction[0,0]: {bp[0,0]}")
print()

print("=" * 60)
print("LEARNING RATE")
print("=" * 60)
print(f"learning_rate: {m.learning_rate}")
print()

print("=" * 60)
print("PREDICTORS STRUCTURE")
print("=" * 60)
print(f"len(_predictors): {len(m._predictors)}")
print(f"_predictors[0] type: {type(m._predictors[0])}")
print(f"_predictors[0] length: {len(m._predictors[0])}")
print(f"_predictors[0][0] type: {type(m._predictors[0][0])}")
print()

print("=" * 60)
print("FIRST PREDICTOR NODE ARRAY")
print("=" * 60)
predictor_0 = m._predictors[0][0]
nodes_0 = predictor_0.nodes
print(f"nodes type: {type(nodes_0)}")
print(f"nodes dtype: {nodes_0.dtype}")
print(f"nodes shape: {nodes_0.shape}")
print(f"nodes dtype fields: {nodes_0.dtype.names}")
print()

print("=" * 60)
print("FIRST 3 NODES OF TREE 0")
print("=" * 60)
for i in range(min(3, len(nodes_0))):
    n = nodes_0[i]
    print(f"  node[{i}]:")
    for field in nodes_0.dtype.names:
        print(f"    {field}: {n[field]}")
    print()

print("=" * 60)
print("FIRST LEAF NODE OF TREE 0")
print("=" * 60)
for i, n in enumerate(nodes_0):
    if n['is_leaf']:
        print(f"  First leaf at index {i}:")
        for field in nodes_0.dtype.names:
            print(f"    {field}: {n[field]}")
        break
print()

print("=" * 60)
print("BIN MAPPER")
print("=" * 60)
bm = m._bin_mapper
print(f"_bin_mapper type: {type(bm)}")
print(f"_bin_mapper.n_bins_no_missing_: {getattr(bm, 'n_bins_no_missing_', 'N/A')}")
bt = bm.bin_thresholds_
print(f"bin_thresholds_ type: {type(bt)}")
print(f"len(bin_thresholds_): {len(bt)}")
for i, (name, arr) in enumerate(zip(FEATURES, bt)):
    print(f"  feature[{i}] {name}: n_thresholds={len(arr)}, min={arr.min():.6g}, max={arr.max():.6g}")
print()

print("=" * 60)
print("MANUAL RAW PREDICTION STEP BY STEP")
print("=" * 60)
# Replicate sklearn's _raw_predict
X_np = X_df.to_numpy(dtype=np.float64)
print(f"Input X shape: {X_np.shape}")

# Try to find sklearn's internal _raw_predict method
print()
print("Checking if _raw_predict exists:")
print(f"  has _raw_predict: {hasattr(m, '_raw_predict')}")

if hasattr(m, '_raw_predict'):
    raw_pred = m._raw_predict(X_np)
    print(f"  _raw_predict result shape: {raw_pred.shape}")
    print(f"  _raw_predict result: {raw_pred}")
print()

print("=" * 60)
print("PREDICTOR predict METHOD INVESTIGATION")
print("=" * 60)
# Check what predict method each predictor has
p0 = m._predictors[0][0]
print(f"predictor type: {type(p0)}")
print(f"predictor methods: {[m_name for m_name in dir(p0) if not m_name.startswith('__')]}")
print()

# Try the predictor's predict directly
try:
    X_binned = m._bin_mapper.transform(X_np)
    print(f"X_binned shape: {X_binned.shape}")
    print(f"X_binned dtype: {X_binned.dtype}")
    print(f"X_binned values: {X_binned}")
    
    pred_from_p0 = p0.predict(X_binned)
    print(f"p0.predict(X_binned): {pred_from_p0}")
except Exception as e:
    print(f"Error calling predictor.predict: {e}")
print()

print("=" * 60)
print("MANUAL TREE TRAVERSAL WITH BINNED FEATURES")
print("=" * 60)
# What does sklearn actually do in the tree traversal?
# HistGradientBoosting works on BINNED features (uint8), not raw float64!
try:
    X_binned = m._bin_mapper.transform(X_np)
    x_binned_row = X_binned[0]
    print(f"x_binned_row: {x_binned_row}")
    print(f"x_binned_row dtype: {x_binned_row.dtype}")
    
    # Now traverse tree 0 using binned feature
    nodes = m._predictors[0][0].nodes
    node_idx = 0
    path = []
    while True:
        node = nodes[node_idx]
        path.append(node_idx)
        if node['is_leaf']:
            print(f"Leaf reached at node {node_idx}, value = {node['value']}")
            break
        feat = int(node['feature_idx'])
        bin_val = x_binned_row[feat]  # binned (uint8)
        threshold = node['num_threshold']  # this is a BIN INDEX, not raw float?
        print(f"  node {node_idx}: feat[{feat}]={bin_val} (binned), threshold={threshold}, left={node['left']}, right={node['right']}")
        if bin_val <= threshold:
            node_idx = int(node['left'])
        else:
            node_idx = int(node['right'])
    print(f"Path taken: {path}")
except Exception as e:
    print(f"Error in binned traversal: {e}")
print()

print("=" * 60)
print("MANUAL ACCUMULATION WITH BINNED FEATURES")
print("=" * 60)
try:
    X_binned = m._bin_mapper.transform(X_np)
    x_binned_row = X_binned[0]
    
    y = float(m._baseline_prediction[0, 0])
    lr = m.learning_rate
    print(f"Starting with baseline: {y}")
    print(f"Learning rate: {lr}")
    
    for tree_idx, predictors_at_stage in enumerate(m._predictors):
        pred_obj = predictors_at_stage[0]
        nodes = pred_obj.nodes
        node_idx = 0
        while True:
            node = nodes[node_idx]
            if node['is_leaf']:
                contrib = lr * float(node['value'])
                y += contrib
                break
            feat = int(node['feature_idx'])
            bin_val = int(x_binned_row[feat])
            threshold = int(node['num_threshold'])
            if np.isnan(float(node['num_threshold'])):
                node_idx = int(node['left']) if node['missing_go_to_left'] else int(node['right'])
            elif bin_val <= threshold:
                node_idx = int(node['left'])
            else:
                node_idx = int(node['right'])
    
    print(f"Manual accumulation result (binned, lr*leaf): {y}")
    print(f"sklearn reference: {sklearn_pred}")
    print(f"Difference: {abs(y - sklearn_pred)}")
except Exception as e:
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("TRY: RAW FLOAT TRAVERSAL WITH lr*leaf (current engine approach)")
print("=" * 60)
try:
    x_raw = np.array(TEST_VECTOR, dtype=np.float64)
    y = float(m._baseline_prediction[0, 0])
    lr = m.learning_rate
    
    for predictors_at_stage in m._predictors:
        nodes = predictors_at_stage[0].nodes
        node_idx = 0
        while True:
            node = nodes[node_idx]
            if node['is_leaf']:
                y += lr * float(node['value'])
                break
            feat = int(node['feature_idx'])
            val = x_raw[feat]
            threshold = float(node['num_threshold'])
            if np.isnan(val):
                node_idx = int(node['left']) if node['missing_go_to_left'] else int(node['right'])
            elif val <= threshold:
                node_idx = int(node['left'])
            else:
                node_idx = int(node['right'])
    
    print(f"Raw float traversal result (lr*leaf): {y}")
    print(f"sklearn reference: {sklearn_pred}")
    print(f"Difference: {abs(y - sklearn_pred)}")
except Exception as e:
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
