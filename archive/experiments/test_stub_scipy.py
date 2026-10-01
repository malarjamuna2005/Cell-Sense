import sys
import types

# Helper to make dummy package
def make_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__file__ = f"<{name}>"
    sys.modules[name] = m
    return m

def make_mod(name):
    m = types.ModuleType(name)
    m.__file__ = f"<{name}>"
    sys.modules[name] = m
    return m

# Create comprehensive dummy module for scipy
def create_scipy_dummies():
    scipy = make_pkg('scipy')
    scipy.__version__ = '1.18.1'

    sparse = make_pkg('scipy.sparse')
    scipy.sparse = sparse
    class DummySparse:
        def __init__(self, *args, **kwargs): pass
        def toarray(self): return np.zeros((1,1))
    sparse.issparse = lambda x: False
    sparse.isspmatrix = lambda x: False
    sparse.isspmatrix_csr = lambda x: False
    sparse.isspmatrix_csc = lambda x: False
    sparse.csr_matrix = DummySparse
    sparse.csc_matrix = DummySparse
    sparse.csr_array = DummySparse
    sparse.csc_array = DummySparse
    sparse.coo_matrix = DummySparse
    sparse.dia_matrix = DummySparse
    sparse.bsr_matrix = DummySparse
    sparse.dok_matrix = DummySparse
    sparse.lil_matrix = DummySparse
    sparse.spmatrix = DummySparse

    sparse_linalg = make_pkg('scipy.sparse.linalg')
    sparse.linalg = sparse_linalg
    sparse_linalg.svds = lambda *a, **kw: None
    sparse_linalg.LinearOperator = DummySparse

    csgraph = make_pkg('scipy.sparse.csgraph')
    sparse.csgraph = csgraph
    csgraph.connected_components = lambda *a, **kw: (1, [0])

    stats = make_pkg('scipy.stats')
    scipy.stats = stats
    stats.mode = lambda a, *args, **kwargs: (a[0] if len(a) else 0, 1)
    stats.rankdata = lambda a, *args, **kwargs: a
    stats.scoreatpercentile = lambda a, per, *args, **kwargs: 0.0

    spatial = make_pkg('scipy.spatial')
    scipy.spatial = spatial
    spatial_dist = make_mod('scipy.spatial.distance')
    spatial.distance = spatial_dist
    spatial_dist.cdist = lambda *a, **kw: None
    spatial_dist.pdist = lambda *a, **kw: None

    opt = make_pkg('scipy.optimize')
    scipy.optimize = opt
    opt.minimize = lambda *a, **kw: None
    opt.curve_fit = lambda *a, **kw: None

    linalg = make_pkg('scipy.linalg')
    scipy.linalg = linalg
    linalg.svd = lambda *a, **kw: None
    linalg.pinv = lambda *a, **kw: None
    linalg.norm = lambda x, *a, **kw: 0.0

    special = make_pkg('scipy.special')
    scipy.special = special
    special.expit = lambda x: 1 / (1 + 2.718281828459045**(-x))
    special.logsumexp = lambda x, *a, **kw: x

create_scipy_dummies()

print("Testing joblib.load with complete scipy package shims...", flush=True)

import sklearn
print(f"sklearn version: {sklearn.__version__}", flush=True)

import joblib
import numpy as np

model_path = 'cellsense_final_soh_model.joblib'
model = joblib.load(model_path)
print("SUCCESSFULLY LOADED MODEL VIA JOBLIB!", flush=True)
print("Model Type:", type(model))
print("Estimator:", model.__class__.__name__)
print("Number of features:", model.n_features_in_)
print("Features:", list(model.feature_names_in_))
print("Params:", model.get_params())
print("Number of iterations:", getattr(model, 'n_iter_', 'N/A'))
print("Baseline prediction:", getattr(model, '_baseline_prediction', 'N/A'))

# Run a test prediction!
# 14 features:
# voltage_mean, voltage_min, voltage_max, voltage_std,
# current_mean, current_min, current_max, current_std,
# discharge_duration_sec, energy_Wh, power_mean, power_std,
# voltage_slope, current_slope
test_sample = np.array([[
    3.75, 3.20, 4.15, 0.18, # voltage stats
    1.25, 0.05, 2.50, 0.45, # current stats
    3540.0,                 # discharge duration
    18.5,                   # energy Wh
    4.68, 1.10,             # power stats
    -0.00015, -0.00008      # slopes
]], dtype=np.float64)

pred = model.predict(test_sample)
print(f"\n================ PREDICTION TEST ================")
print(f"Raw model output: {pred[0]}")
print(f"Formatted SOH: {pred[0]:.2f}%")
print(f"Array shape: {pred.shape}")
print("================================================")
