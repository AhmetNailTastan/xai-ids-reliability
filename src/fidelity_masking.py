from __future__ import annotations

import numpy as np


def mask_top_k_with_means(X, attribution, feature_means, k: int):
    X_masked = np.array(X, copy=True)
    top_features = np.argsort(np.abs(attribution))[-k:]
    X_masked[:, top_features] = np.asarray(feature_means)[top_features]
    return X_masked, top_features


def delta_f1(original_f1: float, masked_f1: float) -> float:
    return float(original_f1 - masked_f1)
