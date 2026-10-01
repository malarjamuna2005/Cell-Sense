import sys
import time

print("Testing direct import of HistGradientBoostingRegressor...", flush=True)

# HistGradientBoosting in sklearn uses numpy and openmp/_hist_gradient_boosting cython extensions
# It doesn't strictly need scipy.spatial._hausdorff!

# Let's inspect sys.modules before
try:
    from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
    print("Success importing HistGradientBoostingRegressor!", flush=True)
except Exception as e:
    print(f"Direct import failed: {e}", flush=True)

try:
    import joblib
    print("joblib version:", joblib.__version__, flush=True)
    
    # Load model with joblib
    model = joblib.load('cellsense_final_soh_model.joblib')
    print("Joblib model loaded successfully!", flush=True)
    print("Model type:", type(model))
    print("Model params:", model.get_params())
    print("feature_names_in_:", model.feature_names_in_)
    print("n_features_in_:", model.n_features_in_)
    print("n_trees_per_iteration_:", getattr(model, 'n_trees_per_iteration_', None))
    print("n_iter_:", getattr(model, 'n_iter_', None))
    print("_baseline_prediction:", getattr(model, '_baseline_prediction', None))
    
    # Test a sample prediction
    import numpy as np
    dummy_input = np.array([[
        3.7, 3.0, 4.2, 0.2, # voltage: mean, min, max, std
        1.5, 0.0, 3.0, 0.5, # current: mean, min, max, std
        3600.0,             # discharge_duration_sec
        15.0,               # energy_Wh
        5.5, 1.2,           # power: mean, std
        -0.0001, -0.0002    # voltage_slope, current_slope
    ]], dtype=np.float64)
    
    pred = model.predict(dummy_input)
    print(f"Sample dummy prediction: {pred} (type: {type(pred)}, value: {pred[0]:.4f})")
except Exception as e:
    import traceback
    traceback.print_exc()
