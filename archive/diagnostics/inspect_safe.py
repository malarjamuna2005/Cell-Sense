import zlib
import pickle
import io
import sys
import pprint

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)
print(f"Decompressed: {len(decompressed)} bytes", flush=True)

class Proxy:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state
    def __repr__(self):
        name = getattr(self, '__class_name__', self.__class__.__name__)
        return f"<Proxy {name} keys={list(self.__dict__.keys())}>"

classes_found = {}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = f"{module}.{name}"
        classes_found[key] = (module, name)
        
        # Only import numpy or builtins
        if module in ('numpy', 'numpy.core.multiarray', 'numpy._core.multiarray', 'builtins'):
            try:
                mod = __import__(module, fromlist=[name])
                return getattr(mod, name)
            except Exception:
                pass

        class DynamicProxy(Proxy):
            pass
        DynamicProxy.__name__ = name
        DynamicProxy.__qualname__ = name
        DynamicProxy.__module__ = module
        DynamicProxy.__class_name__ = key
        return DynamicProxy

data = io.BytesIO(decompressed)
unpickler = SafeUnpickler(data)
obj = unpickler.load()

print("\n================ SUCCESSFUL UNPICKLE ================", flush=True)
print(f"Root object: {obj}", flush=True)
print("\nRoot __dict__ attributes:", flush=True)
for k, v in obj.__dict__.items():
    if not k.startswith('_state'):
        print(f"  {k}: {type(v).__name__} -> {repr(v)[:200]}", flush=True)

print("\nAll classes referenced in model artifact:", flush=True)
for k in sorted(classes_found.keys()):
    print(f"  - {k}", flush=True)
