import zlib
import numpy as np
import io

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)
print(f"Decompressed length: {len(decompressed)} bytes")

# Let's see what numpy arrays are embedded in this joblib file
# In joblib, arrays are serialized using NumpyArrayWrapper or numpy pickle
# Let's parse all objects and arrays

import pickle

# We can unpickle numpy arrays safely with standard pickle if we define a minimal fallback for sklearn classes
class DummyHistGBR:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)

class JoblibArrayWrapper:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def read(self, unpickler):
        return self

# Let's create a custom unpickler that loads all numpy ndarrays and captures the model state
class GBRUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        if 'sklearn' in module:
            class Node:
                def __init__(self, *args, **kwargs):
                    pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                    else:
                        self._state = state
                def __repr__(self):
                    return f"<{name} attrs={list(self.__dict__.keys())}>"
            Node.__name__ = name
            return Node
        return super().find_class(module, name)

try:
    unp = GBRUnpickler(io.BytesIO(decompressed))
    model_obj = unp.load()
    print("SUCCESSFULLY PARSED MODEL OBJECT!")
    print("Attributes in model_obj:")
    for k, v in model_obj.__dict__.items():
        if isinstance(v, np.ndarray):
            print(f"  [Array] {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  [List/Tuple] {k}: len={len(v)}")
            if len(v) > 0:
                print(f"    Item 0 type: {type(v[0])}, preview: {repr(v[0])[:150]}")
        else:
            print(f"  {k}: {repr(v)[:150]}")
except Exception as e:
    import traceback
    traceback.print_exc()
