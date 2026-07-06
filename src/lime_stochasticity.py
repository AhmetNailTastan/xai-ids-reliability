from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .config import LIME_NUM_SAMPLES, N_STOCHASTIC_INSTANCES, N_STOCHASTIC_REPEATS


def lime_vector(exp, n_features: int) -> np.ndarray:
    vector = np.zeros(n_features)
    label = exp.available_labels()[0]
    for feat_idx, value in exp.local_exp[label]:
        vector[int(feat_idx)] = value
    return vector


def compute_lime_stochasticity(
    explainer,
    predict_fn,
    X_samples,
    *,
    n_instances: int = N_STOCHASTIC_INSTANCES,
    n_repeats: int = N_STOCHASTIC_REPEATS,
    num_samples: int = LIME_NUM_SAMPLES,
) -> np.ndarray:
    X_samples = np.asarray(X_samples)[:n_instances]
    n_features = X_samples.shape[1]
    correlations = []

    for x in X_samples:
        vectors = []
        for repeat in range(n_repeats):
            np.random.seed(repeat)
            exp = explainer.explain_instance(
                x,
                predict_fn,
                num_features=n_features,
                num_samples=num_samples,
            )
            vectors.append(lime_vector(exp, n_features))

        for i in range(len(vectors) - 1):
            correlations.append(spearmanr(vectors[i], vectors[i + 1]).correlation)

    return np.asarray(correlations, dtype=float)
