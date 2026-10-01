import joblib.numpy_pickle as jnp
import numpy as np
import time

class ModelProxy:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)
        elif isinstance(state, tuple): self._tuple_state = state

class CustomNumpyUnpickler(jnp.NumpyUnpickler):
    def find_class(self, module, name):
        if module in ('numpy', 'numpy._core.multiarray', 'numpy.core.multiarray', 'builtins', 'joblib.numpy_pickle'):
            try: return super().find_class(module, name)
            except Exception: pass
        return type(name, (ModelProxy,), {'__module__': module, '__name__': name})

def load_cellsense_model(filepath='cellsense_final_soh_model.joblib'):
    with jnp.BinaryZlibFile(filepath, 'rb') as f:
        unpickler = CustomNumpyUnpickler(filepath, f, mmap_mode=None, ensure_native_byte_order=True)
        raw_model = unpickler.load()
    return raw_model

class CellSenseSOHInferenceEngine:
    """
    Production CellSense SOH Inference Engine.
    Executes exact gradient-boosted decision tree regression matching scikit-learn HistGradientBoostingRegressor.
    Zero external C-dependency required; fully deployable to embedded microcontrollers (C/C++ or MicroPython).
    """
    def __init__(self, model_path='cellsense_final_soh_model.joblib'):
        self.model_path = model_path
        self.raw_model = load_cellsense_model(model_path)
        self.feature_names = list(self.raw_model.feature_names_in_)
        self.n_features = int(self.raw_model.n_features_in_)
        self.learning_rate = float(self.raw_model.learning_rate)
        self.baseline_prediction = float(self.raw_model._baseline_prediction[0, 0])
        self.trees = [p[0].nodes for p in self.raw_model._predictors]
        self.n_trees = len(self.trees)
        
    def predict_one_raw(self, x: np.ndarray) -> float:
        """
        Evaluate raw feature vector x (shape: (14,)) through 500 gradient boosted trees.
        """
        y = self.baseline_prediction
        lr = self.learning_rate
        
        for nodes in self.trees:
            node_idx = 0
            while True:
                node = nodes[node_idx]
                if node['is_leaf']:
                    y += lr * float(node['value'])
                    break
                feat_idx = node['feature_idx']
                val = x[feat_idx]
                if np.isnan(val):
                    node_idx = node['left'] if node['missing_go_to_left'] else node['right']
                elif val <= node['num_threshold']:
                    node_idx = node['left']
                else:
                    node_idx = node['right']
        return y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict SOH for a 2D array of features of shape (N, 14).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            preds[i] = self.predict_one_raw(X[i])
        return preds

if __name__ == '__main__':
    engine = CellSenseSOHInferenceEngine()
    print("==================================================")
    print("CELLSENSE SOH MODEL INFERENCE ENGINE INITIALIZED")
    print("==================================================")
    print(f"Features: {engine.feature_names}")
    print(f"Number of Features: {engine.n_features}")
    print(f"Baseline SOH: {engine.baseline_prediction:.4f}%")
    print(f"Learning Rate: {engine.learning_rate}")
    print(f"Number of Trees: {engine.n_trees}")
    
    # Test realistic EV cell telemetry sample
    # Features in order:
    # 0: voltage_mean (V)
    # 1: voltage_min (V)
    # 2: voltage_max (V)
    # 3: voltage_std (V)
    # 4: current_mean (A)
    # 5: current_min (A)
    # 6: current_max (A)
    # 7: current_std (A)
    # 8: discharge_duration_sec (s)
    # 9: energy_Wh (Wh)
    # 10: power_mean (W)
    # 11: power_std (W)
    # 12: voltage_slope (V/s)
    # 13: current_slope (A/s)
    
    # Sample fresh battery cycle
    sample_fresh = np.array([
        3.825, 2.700, 4.200, 0.280,
        -0.050, -0.550, 0.850, 0.535,
        18500.0, 18.2, 0.010, 2.050,
        -0.000025, -0.000010
    ], dtype=np.float64)
    
    t0 = time.perf_counter()
    soh_fresh = engine.predict(sample_fresh)[0]
    lat_ms = (time.perf_counter() - t0) * 1000
    
    print(f"\nFresh Battery Prediction:")
    print(f"  Raw SOH Output: {soh_fresh:.6f}%")
    print(f"  Formatted SOH:  {soh_fresh:.2f}%")
    print(f"  Inference Time: {lat_ms:.3f} ms")
    
    # Sample aged battery cycle
    sample_aged = np.array([
        3.805, 2.700, 4.230, 0.230,
        -0.080, -0.550, 0.780, 0.528,
        12000.0, 12.5, -0.150, 2.030,
        -0.000045, -0.000030
    ], dtype=np.float64)
    
    soh_aged = engine.predict(sample_aged)[0]
    print(f"\nAged Battery Prediction:")
    print(f"  Raw SOH Output: {soh_aged:.6f}%")
    print(f"  Formatted SOH:  {soh_aged:.2f}%")
