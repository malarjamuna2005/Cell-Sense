import zlib
import pickle
import io
import pprint

def inspect_zlib_pickle():
    model_path = r'cellsense_final_soh_model.joblib'
    with open(model_path, 'rb') as f:
        raw = f.read()
        
    decompressed = zlib.decompress(raw)
    print(f"Decompressed length: {len(decompressed)} bytes ({len(decompressed)/1024:.2f} KB)")
    print(f"First 32 decompressed bytes: {decompressed[:32]}")
    
    referenced_classes = set()
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            referenced_classes.add(f"{module}.{name}")
            try:
                mod = __import__(module, fromlist=[name])
                return getattr(mod, name)
            except Exception as e:
                class Dummy:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                        elif isinstance(state, tuple):
                            self._state_tuple = state
                Dummy.__name__ = name
                Dummy.__module__ = module
                return Dummy

    data = io.BytesIO(decompressed)
    unpickler = CustomUnpickler(data)
    res = unpickler.load()
    
    print("\nReferenced Classes in model artifact:")
    for c in sorted(referenced_classes):
        print(f"  - {c}")
        
    print(f"\nRoot object: {type(res)} -> {res}")
    print("\nRoot object attributes:")
    for k in dir(res):
        if not k.startswith('__'):
            try:
                val = getattr(res, k)
                print(f"  {k} ({type(val)}): {repr(val)[:200]}")
            except Exception as e:
                print(f"  {k}: error reading ({e})")
                
    if hasattr(res, '__dict__'):
        print("\nRoot __dict__ keys:")
        for k, v in res.__dict__.items():
            print(f"  {k} ({type(v)}): {repr(v)[:200]}")

if __name__ == '__main__':
    inspect_zlib_pickle()
