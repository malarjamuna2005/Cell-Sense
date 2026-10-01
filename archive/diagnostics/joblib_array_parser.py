import zlib
import numpy as np
import struct
import io
import pickletools

def parse_joblib_file(filepath):
    with open(filepath, 'rb') as f:
        compressed = f.read()
    data = zlib.decompress(compressed)
    print(f"Decompressed size: {len(data)} bytes")
    
    # Let's inspect the exact layout of data
    # Joblib serializes objects using pickle protocol 4 or 5
    # Let's find all NumpyArrayWrappers and reconstructed arrays
    
    # Let's search for strings and objects
    stream = io.BytesIO(data)
    
    # We can use a custom pickle unpickler with a custom find_class
    # where all classes are reconstructed in pure Python
    
    class ModelObject:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self._state = state
        def __repr__(self):
            cls_name = getattr(self, '__class_name__', self.__class__.__name__)
            return f"<{cls_name}>"

    # Joblib NumpyArrayWrapper class
    class NumpyArrayWrapper(ModelObject):
        def read(self, unpickler):
            # In joblib numpy_pickle, NumpyArrayWrapper.read reads array from stream
            # Let's see how joblib reads array
            pass

    import pickle
    
    # Let's build a standalone unpickler that loads all Python types and Numpy arrays
    # In Python 3.12, we can implement the pickle virtual machine or use pickle._Unpickler
    
    # Let's inspect what opcodes are present
    ops = list(pickletools.genops(data))
    print(f"Total opcodes: {len(ops)}")
    
    # Let's see all strings, ints, floats, arrays
    for op, arg, pos in ops:
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE') and any(k in str(arg) for k in ['_baseline', '_bin_mapper', '_predictors', 'bin_thresholds', 'nodes']):
            print(f"  [{pos:6d}] {op.name} -> {arg}")

if __name__ == '__main__':
    parse_joblib_file('cellsense_final_soh_model.joblib')
