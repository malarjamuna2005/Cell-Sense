import numpy as np
from cellsense_engine import CellSenseSOHInferenceEngine

engine = CellSenseSOHInferenceEngine()

# Test different test points to understand model output range
test_points = [
    ("Typical High SOH Test Profile (Long discharge, high energy)", np.array([
        3.84, 2.6995, 4.201, 0.29, -0.02, -0.550, 0.92, 0.54, 800000.0, 18.8, 0.015, 2.08, -3.0e-9, -4.5e-9
    ])),
    ("Mid SOH Test Profile", np.array([
        3.82, 2.6992, 4.205, 0.26, -0.05, -0.550, 0.85, 0.53, 400000.0, 12.0, -0.08, 2.05, -1.0e-6, -2.0e-6
    ])),
    ("Degraded / Low SOH Test Profile (Short discharge, low energy)", np.array([
        3.80, 2.6990, 4.235, 0.22, -0.088, -0.550, 0.78, 0.526, 18000.0, 0.50, -0.21, 2.02, -2.5e-5, -8.5e-5
    ])),
]

print("Model Inference Range Evaluation:")
for label, vec in test_points:
    soh = engine.predict(vec)[0]
    print(f"\n{label}:")
    print(f"  Estimated SOH: {soh:.4f}% ({soh:.2f}%)")
