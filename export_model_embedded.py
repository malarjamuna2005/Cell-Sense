"""
CellSense — Model Export Preparation Layer (Task 9)
=====================================================
Future purpose:
    actual sklearn model
        ↓  (this script)
    verified tree representation
        ↓
    portable intermediate representation  (JSON)
        ↓
    C/C++ arrays                          (NOT yet — pending MCU selection)

This script extracts the complete, verified tree representation from the
trained HistGradientBoostingRegressor and writes it to a portable JSON
file that can be consumed by a future C/C++ code-generator once the actual
MCU has been specified.

IMPORTANT:
- Does NOT modify cellsense_final_soh_model.joblib.
- Does NOT generate MCU firmware yet.
- Does NOT choose an MCU.
- The JSON is an intermediate artefact only.

Verified prediction formula (from validate_engine.py / diagnose_model4.py):

    prediction = baseline_prediction + Σ_{t=0}^{499} traverse_tree(t, x)

where traverse_tree() returns node['value'] at the reached leaf.
node['value'] already encodes  (learning_rate * raw_leaf_contribution).
No additional learning_rate multiplication is needed at inference time.
"""

import os
import json
import joblib
import numpy as np

MODEL_PATH = "cellsense_final_soh_model.joblib"
OUTPUT_DIR = "embedded"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "model_data.json")

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


def float64_to_python(val):
    """Convert numpy float64 to native Python float for JSON serialisation."""
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    return val


def export_tree(nodes):
    """
    Export a single tree's node array to a list of dicts.

    Node fields included (all verified against sklearn 1.6.1 node dtype):
        value           – float64  – lr * raw_leaf_contribution (leaf) or 0 (internal)
        feature_idx     – int64    – which input feature to split on
        num_threshold   – float64  – raw float64 split threshold (raw feature space)
        missing_go_to_left – uint8 – 1 = missing/NaN goes left, 0 = goes right
        left            – uint32   – index of left child node
        right           – uint32   – index of right child node
        is_leaf         – uint8    – 1 = leaf node, 0 = internal node
        bin_threshold   – uint8    – binned threshold (informational; NOT used for inference)

    Inference rule (pure Python / future C):
        node_idx = 0
        while not node[node_idx].is_leaf:
            val = x[node[node_idx].feature_idx]
            if isnan(val):
                node_idx = left if missing_go_to_left else right
            elif val <= node[node_idx].num_threshold:
                node_idx = left
            else:
                node_idx = right
        accumulate += node[node_idx].value
    """
    tree_nodes = []
    for node in nodes:
        tree_nodes.append({
            "value":              float(node["value"]),
            "feature_idx":        int(node["feature_idx"]),
            "num_threshold":      float(node["num_threshold"]),
            "missing_go_to_left": int(node["missing_go_to_left"]),
            "left":               int(node["left"]),
            "right":              int(node["right"]),
            "is_leaf":            int(node["is_leaf"]),
            # bin_threshold is informational — included for diagnostics only
            "bin_threshold":      int(node["bin_threshold"]),
        })
    return tree_nodes


def main():
    print("=" * 60)
    print("CELLSENSE MODEL EXPORT PREPARATION")
    print("=" * 60)
    print()
    print(f"Source model:  {MODEL_PATH}")
    print(f"Output:        {OUTPUT_JSON}")
    print()

    # Load authoritative model
    model = joblib.load(MODEL_PATH)
    print(f"Model class:   {model.__class__.__name__}")

    # Validate structure
    n_trees = len(model._predictors)
    baseline = float(model._baseline_prediction[0, 0])
    lr = float(model.learning_rate)
    n_features = int(model.n_features_in_)

    assert n_trees == 500, f"Expected 500 trees, got {n_trees}"
    assert n_features == 14, f"Expected 14 features, got {n_features}"
    assert list(model.feature_names_in_) == FEATURES, "Feature name mismatch"

    print(f"Trees:         {n_trees}")
    print(f"Features:      {n_features}")
    print(f"Baseline:      {baseline:.15f}")
    print(f"Learning rate: {lr}  (already embedded in leaf values — not needed at inference)")
    print()

    # Build bin_thresholds metadata (float boundaries that define bins)
    bin_thresholds_meta = []
    for feat_name, arr in zip(FEATURES, model._bin_mapper.bin_thresholds_):
        bin_thresholds_meta.append({
            "feature": feat_name,
            "n_thresholds": len(arr),
            "thresholds": [float(v) for v in arr],
        })

    # Export all trees
    print("Exporting trees...")
    trees_export = []
    for i, predictors_at_stage in enumerate(model._predictors):
        predictor = predictors_at_stage[0]
        nodes = predictor.nodes
        trees_export.append({
            "tree_index": i,
            "n_nodes": len(nodes),
            "nodes": export_tree(nodes),
        })
        if (i + 1) % 100 == 0:
            print(f"  Exported {i+1}/{n_trees} trees...")

    total_nodes = sum(t["n_nodes"] for t in trees_export)
    print(f"Total nodes across all trees: {total_nodes}")
    print()

    # Assemble the portable intermediate representation
    export_doc = {
        "schema_version": "1.0",
        "description": (
            "CellSense SOH Model — Portable Intermediate Representation. "
            "Intended for future C/C++ array generation once MCU is specified. "
            "Prediction: baseline + sum(leaf_value for each tree). "
            "lr is embedded in leaf values — do NOT multiply at inference time."
        ),
        "model": {
            "class": model.__class__.__name__,
            "sklearn_version": str(getattr(model, "_sklearn_version", "1.6.1")),
            "n_trees": n_trees,
            "n_features": n_features,
            "feature_names": FEATURES,
            "baseline_prediction": baseline,
            "learning_rate": lr,
            "learning_rate_note": (
                "lr is already encoded in each leaf node['value']. "
                "Inference formula: prediction = baseline + sum(node.value at leaf). "
                "Do NOT multiply node.value by learning_rate at inference time."
            ),
            "loss": str(getattr(model, "loss", "squared_error")),
            "max_leaf_nodes": int(getattr(model, "max_leaf_nodes", 31)),
            "l2_regularization": float(getattr(model, "l2_regularization", 1.0)),
        },
        "inference_formula": {
            "step1": "y = baseline_prediction",
            "step2": "for each tree: traverse using raw float64 x vs num_threshold",
            "step3": "missing/NaN: route via missing_go_to_left",
            "step4": "y += leaf_node.value  (NO additional lr multiplication)",
            "step5": "return y  (SOH %)",
        },
        "node_field_semantics": {
            "value": "lr * raw_leaf_contribution (float64). Already includes lr.",
            "feature_idx": "Index into feature vector (0-13).",
            "num_threshold": "Raw float64 split threshold. Use: val <= num_threshold -> left.",
            "missing_go_to_left": "1=NaN goes left child, 0=NaN goes right child.",
            "left": "Index of left child node.",
            "right": "Index of right child node.",
            "is_leaf": "1=leaf (use value), 0=internal (continue traversal).",
            "bin_threshold": "Informational only. uint8 bin index. NOT used for inference.",
        },
        "bin_mapper_metadata": {
            "description": "Quantile bin boundaries from training. Informational only for export.",
            "missing_values_bin_idx": int(model._bin_mapper.missing_values_bin_idx_),
            "features": bin_thresholds_meta,
        },
        "validation_status": {
            "python_engine_vs_sklearn": "PASS",
            "total_vectors_tested": 215,
            "deterministic_vectors": 15,
            "randomised_vectors_seed42": 200,
            "max_absolute_difference": 0.0,
            "mean_absolute_difference": 0.0,
            "sklearn_version_tested": "1.6.1",
            "note": "Validation performed by validate_engine.py",
        },
        "deployment_status": {
            "python_engine_verified": True,
            "mcu_selected": False,
            "mcu_firmware_generated": False,
            "c_arrays_generated": False,
            "note": (
                "C/C++ array generation is pending MCU specification. "
                "This JSON is the verified intermediate representation."
            ),
        },
        "trees": trees_export,
    }

    # Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export_doc, f, indent=2)

    size_bytes = os.path.getsize(OUTPUT_JSON)
    print(f"Written: {OUTPUT_JSON}")
    print(f"File size: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
    print()
    print("NOTE: This is the Python/JSON intermediate representation.")
    print("      MCU C/C++ arrays are NOT yet generated (MCU not yet specified).")
    print()
    print("Export complete.")


if __name__ == "__main__":
    main()
