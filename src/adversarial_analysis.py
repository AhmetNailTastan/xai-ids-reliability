from __future__ import annotations

import numpy as np


def finite_difference_direction(predict_score_fn, x, *, epsilon=1e-4):
    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)
    base = predict_score_fn(x)
    for idx in range(x.size):
        x_step = x.copy()
        x_step[idx] += epsilon
        gradient[idx] = (predict_score_fn(x_step) - base) / epsilon
    return np.sign(gradient)


def apply_signed_perturbation(X, direction, epsilon: float):
    return np.asarray(X, dtype=float) + epsilon * np.asarray(direction, dtype=float)
