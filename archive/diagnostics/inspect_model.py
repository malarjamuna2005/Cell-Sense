import os
import sys
import pprint
import joblib

def inspect():
    model_path = os.path.join(os.path.dirname(__file__), 'cellsense_final_soh_model.joblib')
    print("==================================================")
    print("CELLSENSE MODEL ARTIFACT INTROSPECTION")
    print("==================================================")
    print(f"Artifact Path: {model_path}")
    if not os.path.exists(model_path):
        print("ERROR: File does not exist!")
        return

    size_bytes = os.path.getsize(model_path)
    print(f"Artifact Size: {size_bytes} bytes ({size_bytes / 1024:.2f} KB / {size_bytes / (1024*1024):.4f} MB)")
    
    obj = joblib.load(model_path)
    print(f"Loaded Object Type: {type(obj)}")
    print(f"Loaded Object Class: {obj.__class__.__name__}")
    print(f"Loaded Object Module: {obj.__class__.__module__}")
    
    if isinstance(obj, dict):
        print("\nDictionary Keys:")
        for k, v in obj.items():
            print(f"  - {k}: {type(v)} -> {repr(v)[:200]}")
    else:
        print("\nObject Details:")
        print(repr(obj))
        
        if hasattr(obj, 'steps'):
            print("\nPipeline Steps:")
            for name, step in obj.steps:
                print(f"  - Step '{name}': {type(step)} -> {step}")
                
        if hasattr(obj, 'feature_names_in_'):
            print("\nfeature_names_in_:")
            for i, feat in enumerate(obj.feature_names_in_):
                print(f"  [{i}] {feat}")
                
        if hasattr(obj, 'n_features_in_'):
            print(f"\nn_features_in_: {obj.n_features_in_}")
            
        if hasattr(obj, 'estimators_'):
            print(f"Estimators count (e.g. ensemble): {len(obj.estimators_)}")
            if len(obj.estimators_) > 0:
                print(f"First estimator type: {type(obj.estimators_[0])}")
                
        if hasattr(obj, 'feature_importances_'):
            print("\nFeature Importances:")
            if hasattr(obj, 'feature_names_in_'):
                for name, imp in zip(obj.feature_names_in_, obj.feature_importances_):
                    print(f"  {name}: {imp:.6f}")
            else:
                for i, imp in enumerate(obj.feature_importances_):
                    print(f"  Feature {i}: {imp:.6f}")

if __name__ == '__main__':
    inspect()
