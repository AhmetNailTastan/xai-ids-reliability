from __future__ import annotations

import numpy as np


def minmax(values):
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if np.isclose(lo, hi):
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def compute_xai_score(perturbation, variation, fidelity, runtime_ms, weights=None):
    weights = weights or {"perturbation": 0.30, "variation": 0.25, "fidelity": 0.30, "runtime": 0.15}
    return (
        weights["perturbation"] * minmax(perturbation)
        + weights["variation"] * minmax(variation)
        + weights["fidelity"] * minmax(fidelity)
        + weights["runtime"] * (1.0 - minmax(runtime_ms))
    )
