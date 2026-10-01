import sys
import types

# Create dummy module for _hausdorff only
if 'scipy.spatial._hausdorff' not in sys.modules:
    dummy_hausdorff = types.ModuleType('scipy.spatial._hausdorff')
    dummy_hausdorff.directed_hausdorff = lambda *args, **kwargs: (0.0, 0, 0)
    sys.modules['scipy.spatial._hausdorff'] = dummy_hausdorff

print("Step 1: importing sklearn...", flush=True)
import sklearn
print(f"Step 2: sklearn version: {sklearn.__version__}", flush=True)

import joblib
print(f"Step 3: joblib version: {joblib.__version__}", flush=True)

model = joblib.load('cellsense_final_soh_model.joblib')
print(f"Step 4: loaded model type: {type(model)}", flush=True)
print(f"Step 5: model representation:\n{model}", flush=True)

if hasattr(model, 'feature_names_in_'):
    print("\nFeature names in:")
    for i, f in enumerate(model.feature_names_in_):
        print(f"  [{i}] {f}")
if hasattr(model, 'n_features_in_'):
    print(f"\nTotal input features: {model.n_features_in_}")
if hasattr(model, 'steps'):
    print(f"\nPipeline Steps: {model.steps}")

print("\nModel attributes:")
for attr in ['estimators_', 'feature_importances_', 'n_iter_', 'train_score_', 'validation_score_']:
    if hasattr(model, attr):
        val = getattr(model, attr)
        if isinstance(val, (list, tuple)):
            print(f"  {attr}: count = {len(val)}")
        else:
            print(f"  {attr}: {val}")
