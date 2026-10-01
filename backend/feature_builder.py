import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

# Exact feature order required by cellsense_final_soh_model.joblib
REQUIRED_FEATURE_ORDER = [
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_min",
    "current_max",
    "current_std",
    "discharge_duration_sec",
    "energy_Wh",
    "power_mean",
    "power_std",
    "voltage_slope",
    "current_slope"
]

# Physical bounds for validation
FEATURE_PHYSICAL_BOUNDS = {
    "voltage_mean": (0.0, 1000.0, "V"),
    "voltage_min": (0.0, 1000.0, "V"),
    "voltage_max": (0.0, 1000.0, "V"),
    "voltage_std": (0.0, 500.0, "V"),
    "current_mean": (-2000.0, 2000.0, "A"),
    "current_min": (-2000.0, 2000.0, "A"),
    "current_max": (-2000.0, 2000.0, "A"),
    "current_std": (0.0, 1000.0, "A"),
    "discharge_duration_sec": (0.0, 10_000_000.0, "s"),
    "energy_Wh": (-100_000.0, 500_000.0, "Wh"),
    "power_mean": (-1_000_000.0, 1_000_000.0, "W"),
    "power_std": (0.0, 1_000_000.0, "W"),
    "voltage_slope": (-100.0, 100.0, "V/s"),
    "current_slope": (-100.0, 100.0, "A/s"),
}

class FeatureValidationError(ValueError):
    """Raised when incoming telemetry fails feature validation."""
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors

def build_features_from_dict(telemetry_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds the 14-element feature vector from a dictionary of pre-aggregated or manual features.
    Strictly preserves the exact feature ordering of the trained model.
    """
    errors = []
    values = []
    parsed_features = {}

    for name in REQUIRED_FEATURE_ORDER:
        if name not in telemetry_dict:
            errors.append(f"Missing required feature: '{name}'")
            continue
        
        raw_val = telemetry_dict[name]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Feature '{name}' has invalid non-numeric value: {repr(raw_val)}")
            continue

        if np.isnan(val):
            errors.append(f"Feature '{name}' cannot be NaN")
            continue
        if np.isinf(val):
            errors.append(f"Feature '{name}' cannot be Infinite")
            continue

        # Physical bounds check
        if name in FEATURE_PHYSICAL_BOUNDS:
            min_val, max_val, unit = FEATURE_PHYSICAL_BOUNDS[name]
            if not (min_val <= val <= max_val):
                errors.append(f"Feature '{name}' value {val} {unit} is outside physically possible bounds [{min_val}, {max_val}] {unit}")

        values.append(val)
        parsed_features[name] = val

    if errors:
        raise FeatureValidationError(errors)

    feature_array = np.array(values, dtype=np.float64)
    validation_report = {
        "is_valid": True,
        "feature_count": len(feature_array),
        "feature_names": REQUIRED_FEATURE_ORDER,
        "parsed_values": parsed_features
    }
    return feature_array, validation_report

def extract_features_from_timeseries(
    voltages: List[float],
    currents: List[float],
    timestamps: Optional[List[float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts the 14 model features from raw time-series measurements.
    Computes statistical properties, energy, and linear trend slopes.
    """
    if len(voltages) == 0 or len(currents) == 0:
        raise FeatureValidationError(["Time series contains no data points"])
    if len(voltages) != len(currents):
        raise FeatureValidationError(["Voltages and currents arrays must have equal length"])
    if len(voltages) < 2:
        raise FeatureValidationError(["At least 2 time-series samples are required to extract cycle statistics and slopes"])

    v_arr = np.array(voltages, dtype=np.float64)
    i_arr = np.array(currents, dtype=np.float64)

    if np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)):
        raise FeatureValidationError(["Voltage time-series contains NaN or Infinite values"])
    if np.any(np.isnan(i_arr)) or np.any(np.isinf(i_arr)):
        raise FeatureValidationError(["Current time-series contains NaN or Infinite values"])

    # Timestamps in seconds
    if timestamps is None or len(timestamps) != len(voltages):
        t_arr = np.arange(len(voltages), dtype=np.float64)
    else:
        t_arr = np.array(timestamps, dtype=np.float64)

    dt = np.diff(t_arr)
    duration_sec = float(t_arr[-1] - t_arr[0])
    if duration_sec <= 0:
        duration_sec = float(len(voltages))

    # Instantaneous power in Watts
    power_arr = v_arr * i_arr
    
    # Energy in Watt-hours
    if len(dt) > 0 and np.all(dt > 0):
        # Trapezoidal integration for energy
        energy_Wh = float(np.sum(0.5 * (power_arr[:-1] + power_arr[1:]) * dt) / 3600.0)
    else:
        energy_Wh = float(np.mean(power_arr) * (duration_sec / 3600.0))

    # Linear slope estimation (least squares)
    if duration_sec > 0:
        t_norm = t_arr - t_arr[0]
        v_slope, _ = np.polyfit(t_norm, v_arr, 1)
        i_slope, _ = np.polyfit(t_norm, i_arr, 1)
    else:
        v_slope = 0.0
        i_slope = 0.0

    features_dict = {
        "voltage_mean": float(np.mean(v_arr)),
        "voltage_min": float(np.min(v_arr)),
        "voltage_max": float(np.max(v_arr)),
        "voltage_std": float(np.std(v_arr)),
        "current_mean": float(np.mean(i_arr)),
        "current_min": float(np.min(i_arr)),
        "current_max": float(np.max(i_arr)),
        "current_std": float(np.std(i_arr)),
        "discharge_duration_sec": duration_sec,
        "energy_Wh": energy_Wh,
        "power_mean": float(np.mean(power_arr)),
        "power_std": float(np.std(power_arr)),
        "voltage_slope": float(v_slope),
        "current_slope": float(i_slope)
    }

    return build_features_from_dict(features_dict)
