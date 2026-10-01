"""
CellSense - Deep Diagnostic Part 3
Read the ACTUAL sklearn source files to understand what _predict_iterations does.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
import os
import inspect

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

# Read the gradient_boosting source
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb_module

print("=" * 60)
print("_predict_iterations SOURCE:")
print("=" * 60)
try:
    src = inspect.getsource(gb_module.BaseHistGradientBoosting._predict_iterations)
    print(src)
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("predict SOURCE (HistGradientBoostingRegressor):")
print("=" * 60)
try:
    from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
    src = inspect.getsource(HistGradientBoostingRegressor.predict)
    print(src)
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("TreePredictor source file:")
print("=" * 60)
from sklearn.ensemble._hist_gradient_boosting import predictor
print(predictor.__file__)

# Read the actual predictor.py file
predictor_src_path = predictor.__file__
# It might be a .pyx compiled file - let's check
if predictor_src_path.endswith('.pyc'):
    predictor_src_path = predictor_src_path[:-1]  # try .py

print(f"Attempting to read: {predictor_src_path}")
if os.path.exists(predictor_src_path):
    with open(predictor_src_path, 'r') as f:
        print(f.read())
else:
    # It's a compiled Cython module - look for .pyx
    pyx_path = predictor_src_path.replace('.pyd', '.pyx').replace('.so', '.pyx')
    sklearn_dir = os.path.dirname(os.path.dirname(predictor_src_path))
    print(f"Compiled module - looking for .pyx in {os.path.dirname(predictor_src_path)}")
    for fn in os.listdir(os.path.dirname(predictor_src_path)):
        print(f"  {fn}")

print()
print("=" * 60)
print("CHECKING _preprocess_X:")
print("=" * 60)
try:
    src = inspect.getsource(gb_module.BaseHistGradientBoosting._preprocess_X)
    print(src)
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("Full gradient_boosting module source file location:")
print("=" * 60)
print(gb_module.__file__)

print()
print("=" * 60)
print("INVESTIGATING _predict_iterations with minimal call:")
print("=" * 60)
# What does _predict_iterations actually do?
# According to the source, it calls _predict_iterations which then uses C extension
# Let's trace what happens to X before prediction
X_np = X_df.to_numpy(dtype=np.float64)
print(f"X_np dtype: {X_np.dtype}")
print(f"X_np: {X_np}")

# Try calling _preprocess_X to see what comes out
try:
    X_preprocessed = m._preprocess_X(X_np, reset=False)
    print(f"_preprocess_X output dtype: {X_preprocessed.dtype}")
    print(f"_preprocess_X output: {X_preprocessed}")
except Exception as e:
    print(f"Error calling _preprocess_X: {e}")

# After preprocessing, does it call _bin_mapper.transform?
print()
print("=" * 60)
print("CHECKING: Does predict go through binning or not?")
print("=" * 60)
# The _raw_predict source shows: is_binned = getattr(self, "_in_fit", False)
# When _in_fit is False (prediction time), it calls _preprocess_X
# Let's see what _preprocess_X does

# Let's check _predict_iterations signature
print("Checking _predict_iterations in base class...")
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as gb
src_file = gb.__file__
print(f"Module file: {src_file}")
with open(src_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find _predict_iterations
start = content.find('def _predict_iterations')
if start != -1:
    end = content.find('\n    def ', start + 1)
    print(content[start:end if end != -1 else start+3000])
else:
    print("_predict_iterations not found in .py file - may be in C extension")
    
# Find _preprocess_X  
start = content.find('def _preprocess_X')
if start != -1:
    end = content.find('\n    def ', start + 1)
    print("\n_preprocess_X:")
    print(content[start:end if end != -1 else start+3000])
