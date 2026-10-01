"""
CellSense — Application Smoke Test
Tests all API endpoints and confirms the full prediction flow works.
"""
import urllib.request
import json
import sys

BASE = "http://127.0.0.1:8000"
results = []

def get(path, label):
    url = BASE + path
    try:
        r = urllib.request.urlopen(url, timeout=5)
        data = json.loads(r.read())
        results.append((True, label, r.status, data))
        return True, r.status, data
    except urllib.error.HTTPError as e:
        results.append((False, label, e.code, str(e)))
        return False, e.code, {}
    except Exception as e:
        results.append((False, label, None, str(e)))
        return False, None, {}

def post(path, payload, label):
    url = BASE + path
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=10)
        data = json.loads(r.read())
        results.append((True, label, r.status, data))
        return True, r.status, data
    except urllib.error.HTTPError as e:
        results.append((False, label, e.code, str(e)))
        return False, e.code, {}
    except Exception as e:
        results.append((False, label, None, str(e)))
        return False, None, {}

print("=" * 60)
print("CELLSENSE APPLICATION SMOKE TEST")
print("=" * 60)

# A. Health
ok, status, d = get("/api/health", "A. Backend health")
if ok:
    print(f"  PASS  A. Backend health        HTTP {status}  model_loaded={d.get('model_loaded')}")
else:
    print(f"  FAIL  A. Backend health        HTTP {status}  {d}")

# B. Frontend
url = BASE + "/"
try:
    r = urllib.request.urlopen(url, timeout=5)
    html = r.read().decode("utf-8", errors="replace")
    has_cs = "CELLSENSE" in html.upper()
    has_soh = "Estimated SOH" in html or "State of Health" in html
    print(f"  {'PASS' if has_cs else 'FAIL'}  B. Frontend loads          HTTP {r.status}  CELLSENSE={has_cs}  SOH_label={has_soh}")
except Exception as e:
    print(f"  FAIL  B. Frontend loads          {e}")

# C. Model loads
ok, status, d = get("/api/ml/model", "C. Model info")
if ok:
    mtype = d.get("model_type", "?")
    nfeat = d.get("input_feature_count", "?")
    mstat = d.get("status", "?")
    print(f"  PASS  C. Model info            HTTP {status}  type={mtype}  features={nfeat}  status={mstat}")
else:
    print(f"  FAIL  C. Model info            HTTP {status}  {d}")

# D. Prediction works (canonical test vector)
CANON_TELEMETRY = {
    "voltage_mean": 3.82, "voltage_min": 2.70, "voltage_max": 4.20, "voltage_std": 0.25,
    "current_mean": -0.05, "current_min": -0.55, "current_max": 0.85, "current_std": 0.53,
    "discharge_duration_sec": 50000, "energy_Wh": 10.0,
    "power_mean": -0.10, "power_std": 2.05,
    "voltage_slope": -0.00001, "current_slope": -0.00002,
}
ok, status, d = post("/api/ml/predict", {"telemetry": CANON_TELEMETRY}, "D. Predict")
if ok:
    soh = d.get("soh")
    lat = d.get("inference_latency_ms")
    src = d.get("source")
    ist = d.get("inference_status")
    print(f"  PASS  D. Prediction works      HTTP {status}  SOH={soh}%  latency={lat}ms  source={src}  status={ist}")
else:
    print(f"  FAIL  D. Prediction works      HTTP {status}  {d}")

# E. Missing features rejected (422)
ok, status, d = post("/api/ml/predict", {"telemetry": {"voltage_mean": 3.8}}, "E. Missing features")
rejected = not ok and status == 422
print(f"  {'PASS' if rejected else 'FAIL'}  E. Missing features rejected  HTTP {status}")

# F. Demo start
ok, status, d = post("/api/demo/start", {}, "F. Demo start")
if ok:
    print(f"  PASS  F. Demo starts           HTTP {status}  demo_mode={d.get('demo_mode')}")
else:
    print(f"  FAIL  F. Demo starts           HTTP {status}")

# G. Telemetry with demo active
ok, status, d = get("/api/telemetry", "G. Telemetry")
print(f"  {'PASS' if ok else 'FAIL'}  G. Telemetry endpoint       HTTP {status}")

# H. History
ok, status, d = get("/api/ml/history", "H. History")
if ok:
    cnt = d.get("count", 0)
    print(f"  PASS  H. History               HTTP {status}  count={cnt}")
else:
    print(f"  FAIL  H. History               HTTP {status}")

# I. Battery status
ok, status, d = get("/api/battery/status", "I. Battery status")
if ok:
    print(f"  PASS  I. Battery status        HTTP {status}  soh={d.get('soh')}  count={d.get('prediction_count')}")
else:
    print(f"  FAIL  I. Battery status        HTTP {status}")

# J. Alerts
ok, status, d = get("/api/alerts", "J. Alerts")
print(f"  {'PASS' if ok else 'FAIL'}  J. Alerts endpoint          HTTP {status}")

# K. Demo stop
ok, status, d = post("/api/demo/stop", {}, "K. Demo stop")
if ok:
    print(f"  PASS  K. Demo stops            HTTP {status}  demo_mode={d.get('demo_mode')}")
else:
    print(f"  FAIL  K. Demo stops            HTTP {status}")

# L. Invalid input (NaN-encoded value — caught by validator)
ok, status, d = post("/api/ml/predict",
    {"telemetry": {k: (float("nan") if k == "voltage_mean" else 1.0) for k in CANON_TELEMETRY}},
    "L. Invalid (NaN) input")
rejected = not ok and status == 422
print(f"  {'PASS' if rejected else 'FAIL'}  L. NaN input rejected        HTTP {status}")

# Summary
print()
print("=" * 60)
total = 12
failures = [r for r in results if not r[0]]
passed = total - len(failures)
print(f"SMOKE TEST RESULT:  {passed}/{total} checks passed")
if failures:
    print("Failures:")
    for _, lbl, st, msg in failures:
        print(f"  - {lbl}: HTTP {st}  {str(msg)[:80]}")
else:
    print("All smoke test checks passed.")
print("=" * 60)
