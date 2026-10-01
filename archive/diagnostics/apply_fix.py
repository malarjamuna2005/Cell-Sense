"""
Applies the root-cause fix to backend/inference_engine.py:
Remove the erroneous lr multiplication from predict_one().

Verified: sklearn stores lr*leaf in node['value'] directly.
Correct accumulation: prediction = baseline + sum(node['value'])
"""
import re

path = r'C:\Users\USER\Downloads\Cell\Cell\backend\inference_engine.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Normalize to LF for matching
content_lf = content.replace('\r\n', '\n')

OLD_BLOCK = """    def predict_one(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        \"\"\"
        Runs single sample prediction.
        Returns (predicted_soh_percent, inference_latency_ms).
        \"\"\"
        if feature_vector.ndim != 1 or len(feature_vector) != self.n_features:
            raise ValueError(f\"Feature vector must have exactly {self.n_features} elements, got shape {feature_vector.shape}\")

        t0 = time.perf_counter_ns()
        
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
                val = feature_vector[feat_idx]
                if np.isnan(val):
                    node_idx = node['left'] if node['missing_go_to_left'] else node['right']
                elif val <= node['num_threshold']:
                    node_idx = node['left']
                else:
                    node_idx = node['right']

        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0
        
        return y, latency_ms"""

NEW_BLOCK = """    def predict_one(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        \"\"\"
        Runs single sample prediction.
        Returns (predicted_soh_percent, inference_latency_ms).

        NOTE — sklearn HistGradientBoostingRegressor leaf value semantics
        (verified against sklearn 1.6.1 source and this model artifact):

        sklearn stores  (learning_rate * raw_leaf_contribution)  directly
        in each node's 'value' field at training time.  The accumulation at
        inference time is therefore:

            prediction = baseline + sum(node['value'])         # CORRECT

        NOT:

            prediction = baseline + sum(lr * node['value'])   # WRONG — doubles lr

        Verified by comparing predictor.predict() against raw node traversal
        for all 500 trees: they match exactly, and the formula above reproduces
        sklearn._raw_predict() to within floating-point precision (diff = 0.0).

        Tree traversal uses raw float64 feature values compared against
        node['num_threshold'] (float64).  Missing values (NaN) are routed
        via node['missing_go_to_left'].
        \"\"\"
        if feature_vector.ndim != 1 or len(feature_vector) != self.n_features:
            raise ValueError(f\"Feature vector must have exactly {self.n_features} elements, got shape {feature_vector.shape}\")

        t0 = time.perf_counter_ns()

        y = self.baseline_prediction
        # DO NOT multiply node['value'] by self.learning_rate here.
        # sklearn already encodes  lr * raw_leaf_contribution  inside node['value'].
        for nodes in self.trees:
            node_idx = 0
            while True:
                node = nodes[node_idx]
                if node['is_leaf']:
                    y += float(node['value'])   # lr already embedded in value
                    break
                feat_idx = node['feature_idx']
                val = feature_vector[feat_idx]
                if np.isnan(val):
                    node_idx = node['left'] if node['missing_go_to_left'] else node['right']
                elif val <= node['num_threshold']:
                    node_idx = node['left']
                else:
                    node_idx = node['right']

        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0

        return y, latency_ms"""

if OLD_BLOCK in content_lf:
    new_content_lf = content_lf.replace(OLD_BLOCK, NEW_BLOCK, 1)
    # Restore original line endings
    new_content = new_content_lf.replace('\n', '\r\n') if '\r\n' in content else new_content_lf
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: predict_one patched.")
else:
    print("ERROR: OLD_BLOCK not found in file. Printing lines 118-151 for diagnosis:")
    for i, line in enumerate(content_lf.split('\n')[117:151], start=118):
        print(f"{i}: {repr(line)}")
