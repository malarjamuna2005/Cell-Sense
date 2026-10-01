import joblib.numpy_pickle as jnp
import numpy as np

class ModelProxy:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            self._tuple_state = state
    def __repr__(self):
        return f"<{self.__class__.__name__} keys={list(self.__dict__.keys())}>"

class CustomNumpyUnpickler(jnp.NumpyUnpickler):
    def find_class(self, module, name):
        # Only import numpy or builtins, proxy all sklearn/scipy/_loss classes
        if module in ('numpy', 'numpy._core.multiarray', 'numpy.core.multiarray', 'builtins', 'joblib.numpy_pickle'):
            try:
                return super().find_class(module, name)
            except Exception:
                pass
        proxy_cls = type(name, (ModelProxy,), {'__module__': module, '__name__': name})
        return proxy_cls

filename = 'cellsense_final_soh_model.joblib'
with jnp.BinaryZlibFile(filename, 'rb') as f:
    unpickler = CustomNumpyUnpickler(filename, f, mmap_mode=None, ensure_native_byte_order=True)
    model = unpickler.load()

print("==================================================")
print("SUCCESSFULLY LOADED MODEL VIA CUSTOM UNPICKLER!")
print("==================================================")
print("Model object:", model)
print("\nModel Attributes & Parameters:")
for k, v in model.__dict__.items():
    if isinstance(v, np.ndarray):
        print(f"  [ndarray] {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"  [list/tuple] {k}: len={len(v)}")
        if len(v) > 0:
            print(f"    [0]: {type(v[0])} -> {repr(v[0])[:120]}")
    else:
        print(f"  {k}: {repr(v)[:120]}")

print("\n--- Bin Mapper Details ---")
if hasattr(model, '_bin_mapper'):
    bm = model._bin_mapper
    print(f"BinMapper attrs: {list(bm.__dict__.keys())}")
    if hasattr(bm, 'bin_thresholds_'):
        bt = bm.bin_thresholds_
        print(f"bin_thresholds_ count: {len(bt)}")
        for i, arr in enumerate(bt):
            print(f"  Feature {i:2d} ({model.feature_names_in_[i]}): bins={len(arr)}, min={arr.min():.4f}, max={arr.max():.4f}")

print("\n--- Predictors / Decision Trees ---")
if hasattr(model, '_predictors'):
    preds = model._predictors
    print(f"Total boosting iterations (trees): {len(preds)}")
    if len(preds) > 0:
        first_iter = preds[0]
        tree = first_iter[0]
        print(f"Iteration 0 Tree: {tree}")
        nodes = tree.nodes
        print(f"  nodes.shape: {nodes.shape}, dtype: {nodes.dtype}")
        print(f"  fields: {nodes.dtype.names}")
        print(f"  Total nodes in tree 0: {len(nodes)}")
        for idx in range(min(5, len(nodes))):
            print(f"    Node {idx}: {nodes[idx]}")
