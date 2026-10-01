import numpy as np
from cellsense_engine import CellSenseSOHInferenceEngine

engine = CellSenseSOHInferenceEngine()

print("Model Inspection Summary:")
print("Features & Threshold ranges:")
for i, name in enumerate(engine.feature_names):
    bt = engine.raw_model._bin_mapper.bin_thresholds_[i]
    print(f"  {i:2d}: {name:<24} | Bins: {len(bt):2d} | Range: [{bt.min() if len(bt) else 'N/A'}, {bt.max() if len(bt) else 'N/A'}]")

# Let's inspect leaf values across all trees
all_leaf_values = []
feature_usage_count = {i: 0 for i in range(engine.n_features)}

for nodes in engine.trees:
    for node in nodes:
        if node['is_leaf']:
            all_leaf_values.append(node['value'])
        else:
            feature_usage_count[node['feature_idx']] += 1

all_leaf_values = np.array(all_leaf_values)
print("\nTree Ensemble Stats:")
print(f"Total Trees: {len(engine.trees)}")
print(f"Total Leaf Nodes: {len(all_leaf_values)}")
print(f"Leaf Value Range: [{all_leaf_values.min():.4f}, {all_leaf_values.max():.4f}]")
print(f"Baseline value: {engine.baseline_prediction:.4f}")

print("\nFeature Split Frequency (Feature Importance Proxy):")
total_splits = sum(feature_usage_count.values())
for i, name in enumerate(engine.feature_names):
    count = feature_usage_count[i]
    pct = (count / total_splits) * 100 if total_splits > 0 else 0
    print(f"  {name:<24} : {count:4d} splits ({pct:5.1f}%)")
