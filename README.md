# CellSense — EV Battery Health Monitoring System
### Real-Time SOH Inference & Gateway Intelligence Platform

CellSense is an aftermarket Electric Vehicle (EV) battery health monitoring system designed to estimate battery State of Health (SOH) in real time from 14 cycle-derived electrical features using a gradient-boosted decision tree ensemble.

---

## 1. System Architecture

```
EV Battery / BMS → CAN / OBD-II
            ↓
  CellSense Device / Gateway
            ↓
  Telemetry Acquisition & 14-Feature Extraction
            ↓
  On-Device / Embedded Inference Engine
  (HistGradientBoostingRegressor — 500 Trees)
            ↓
  Real-Time SOH Estimation
            ↓
  BLE / Wi-Fi / WebSocket Transport
            ↓
  CellSense Reactive Web / Mobile Dashboard
```

### Architectural Decoupling
The user interface communicates exclusively through the `DataSource` interface (`PythonModelDataSource` vs `MCUDataSource`). Migrating inference from the local Python gateway to an on-device embedded MCU requires updating only the hardware transport adapter without modifying the dashboard UI.

---

## 2. Repository Structure

```
Cell/
├── backend/
│   ├── __init__.py               # Package marker
│   ├── main.py                   # FastAPI application (REST API, WebSocket telemetry, static UI)
│   ├── inference_engine.py       # Validated embedded-style tree traversal inference engine
│   ├── feature_builder.py        # 14-feature parser, validation, and physical bounds checking
│   └── data_source.py            # DataSource abstraction (PythonModel, Mock/Demo, MCU stub)
├── frontend/
│   └── index.html                # Responsive web dashboard with connection loading screen (WCAG 2.1 AA)
├── tests/
│   ├── __init__.py               # Test package marker
│   ├── test_api.py               # REST API and WebSocket integration tests
│   ├── test_model.py             # Model loading, feature order, and validation tests
│   └── test_engine_equivalence.py# 25 equivalence tests comparing engine vs scikit-learn
├── embedded/
│   └── model_data.json           # 4.51 MB JSON intermediate tree export (generated artifact)
├── archive/                      # Preserved historical diagnostic and exploration scripts
│   ├── diagnostics/              # One-off model inspection and diagnostic scripts
│   └── experiments/              # Legacy standalone prototypes and exploration scripts
├── cellsense_final_soh_model.joblib # Authoritative trained model artifact (DO NOT MODIFY)
├── export_model_embedded.py      # Standalone script to export model trees to portable JSON
├── validate_engine.py            # 215-vector equivalence validation suite (0.0 diff verification)
├── requirements.txt              # Pinned Python dependencies
├── .gitignore                    # Standard repository ignore rules
├── RUNTIME_CONNECTION_FIX_REPORT.md # Historical runtime fix report
└── README.md                     # Project documentation
```

---

## 3. Installation & Prerequisites

### Prerequisites
- **Python**: 3.11 (tested on Python 3.11.9)
- **pip**: >= 24.0

### Setup Environment
```bash
# Clone or navigate to the repository
cd Cell

# Optional: Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

---

## 4. Running the Application

### 1. Start the Backend Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 2. Access the Dashboard
Open your web browser and navigate to:
- **Dashboard**: [http://localhost:8000/](http://localhost:8000/) or [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

*(Note: `0.0.0.0` is the server bind address; use `localhost` or `127.0.0.1` in the browser).*

---

## 5. API Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the CellSense reactive dashboard (`frontend/index.html`) |
| `GET` | `/api/health` | Service health, uptime, and model load status |
| `GET` | `/api/device/status` | Active gateway connection type and hardware status |
| `GET` | `/api/ml/model` | Authentic ML model metadata, architecture, and hyperparameters |
| `POST` | `/api/ml/predict` | Computes SOH from a 14-feature dictionary payload |
| `GET` | `/api/ml/history` | Historical prediction log (bounded to recent 500 items) |
| `GET` | `/api/battery/status`| Latest SOH prediction and system readiness status |
| `GET` | `/api/alerts` | Active system and battery condition alerts |
| `POST` | `/api/demo/start` | Initiates background demo telemetry stream |
| `POST` | `/api/demo/stop` | Stops background demo telemetry stream |
| `WS` | `/ws/telemetry` | Real-time WebSocket stream for telemetry, predictions, and control |

---

## 6. Authoritative ML Model & Feature Pipeline

### Model Specification
- **Model Type**: `HistGradientBoostingRegressor`
- **Framework**: `scikit-learn` v1.6.1
- **Artifact File**: `cellsense_final_soh_model.joblib` (243,766 bytes / ~238 KB)
- **Target Variable**: `SOH_percent` (Continuous numeric regression)
- **Loss Function**: `squared_error` (`HalfSquaredError`)
- **Ensemble Size**: 500 boosting iterations (500 decision trees)
- **Max Leaf Nodes**: 31 leaves per tree
- **Learning Rate**: 0.05 (embedded directly in leaf values at training time)
- **Baseline Prediction**: 37.66632221833227

### Strict 14-Feature Input Order
All predictions require exactly 14 numerical features in this exact sequence:

| Index | Feature Name | Description | Physical Unit | Bounds |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `voltage_mean` | Cycle average cell voltage | V | [0.0, 1000.0] |
| 1 | `voltage_min` | Minimum discharge voltage | V | [0.0, 1000.0] |
| 2 | `voltage_max` | Maximum charging/cutoff voltage | V | [0.0, 1000.0] |
| 3 | `voltage_std` | Standard deviation of voltage | V | [0.0, 500.0] |
| 4 | `current_mean` | Cycle average current | A | [-2000.0, 2000.0] |
| 5 | `current_min` | Minimum discharge current | A | [-2000.0, 2000.0] |
| 6 | `current_max` | Maximum charge current | A | [-2000.0, 2000.0] |
| 7 | `current_std` | Standard deviation of current | A | [0.0, 1000.0] |
| 8 | `discharge_duration_sec` | Total discharge duration | s | [0.0, 10,000,000.0] |
| 9 | `energy_Wh` | Net discharged energy | Wh | [-100,000.0, 500,000.0] |
| 10 | `power_mean` | Cycle average power | W | [-1,000,000.0, 1,000,000.0] |
| 11 | `power_std` | Standard deviation of power | W | [0.0, 1,000,000.0] |
| 12 | `voltage_slope` | Voltage discharge slope ($dV/dt$) | V/s | [-100.0, 100.0] |
| 13 | `current_slope` | Current discharge slope ($dI/dt$) | A/s | [-100.0, 100.0] |

---

## 7. Model Validation & Equivalence Results

The production inference engine (`backend/inference_engine.py`) performs native decision tree traversal over the 500 boosting trees, matching scikit-learn's C-extension implementation to floating-point precision.

### Verification Summary
Run the comprehensive validation suite:
```bash
python validate_engine.py
```

| Validation Suite | Vectors Tested | Passed | Failed | Max Abs Difference | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reference Deterministic Vector | 1 | 1 | 0 | `0.000e+00` | **PASS** |
| Deterministic Multi-Region Vectors | 15 | 15 | 0 | `0.000e+00` | **PASS** |
| Randomized Distribution Vectors (seed=42) | 200 | 200 | 0 | `0.000e+00` | **PASS** |
| **Total Validation** | **215** | **215** | **0** | **`0.000e+00`** | **PASS** |

### Automated Test Suite
Run the full pytest suite:
```bash
python -m pytest -v
```
**Result**: 56 passed / 56 total (100% PASS).

---

## 8. Embedded Export Workflow

To generate a portable JSON intermediate representation of all 500 decision trees for future C/C++ MCU firmware compilation:
```bash
python export_model_embedded.py
```
- **Output**: `embedded/model_data.json` (4.51 MB)
- **Content**: 16,434 nodes across 500 trees with float64 thresholds, left/right child indices, leaf contributions, and missing-value routing flags.

---

## 9. Dataset & Training Lineage

- **Training Datasets**: NASA Ames Battery Aging Datasets & CALCE CS2 Battery Cycle Datasets.
- **Feature Lineage Standard**: The authoritative feature pipeline follows the verified evolution:
  $$\text{stage\_8O53} \longrightarrow \text{stage\_8O69} \longrightarrow \text{stage\_8O71} \longrightarrow \text{stage\_8O86}$$
- **Temperature Rise Definition**: The authoritative temperature rise calculation used during feature engineering is defined as:
  $$\text{temperature\_rise} = \text{temperature\_max} - \text{temperature\_min}$$
  *(Note: `stage_8O68` is explicitly deprecated and excluded from authoritative lineage due to a constant 1.0 temperature rise artifact).*

---

## 10. Repository Hygiene & Archive Notes

To maintain a clean production repository:
- All one-off diagnostic and reverse-engineering scripts are preserved under `archive/diagnostics/`.
- Legacy standalone prototypes are preserved under `archive/experiments/`.
- The authoritative production engine resides exclusively at `backend/inference_engine.py`.

---

## 11. Deployment Status & Claim Safety

- **Verified**: Python scikit-learn model, Python embedded-style engine, REST API, WebSocket streaming, reactive web dashboard, 215/215 numerical equivalence.
- **Designed/Proposed**: On-device MCU inference (firmware generation and hardware benchmarking pending specific MCU target selection).
- **Honesty Standard**: Inference latencies reported in the UI are measured on the local host and are not represented as MCU hardware metrics.

---

## 12. TODO — License / Intellectual Property

> **Notice**: Licensing and public release terms are pending project-owner decision. All rights reserved. Do not distribute or redistribute without explicit authorization.
