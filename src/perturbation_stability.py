from __future__ import annotations

import numpy as np

from .utils import jaccard_top_k, safe_spearman


def perturb_inputs(X, noise_ratio: float, *, rng=None):
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    scale = np.nanstd(X, axis=0)
    return X + rng.normal(0.0, noise_ratio * scale, size=X.shape)


def compare_attributions(original, perturbed, *, top_k=(5, 10)) -> dict:
    record = {"spearman": safe_spearman(original, perturbed)}
    for k in top_k:
        record[f"jaccard_{k}"] = jaccard_top_k(original, perturbed, k)
    return record
