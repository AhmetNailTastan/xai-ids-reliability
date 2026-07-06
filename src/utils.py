from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_pickle(path: str | Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def dump_pickle(obj, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as fh:
        pickle.dump(obj, fh)


def jaccard_top_k(a, b, k: int) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if k <= 0:
        raise ValueError("k must be positive")
    top_a = set(np.argsort(np.abs(a))[-k:])
    top_b = set(np.argsort(np.abs(b))[-k:])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else np.nan


def safe_spearman(a, b) -> float:
    value = spearmanr(a, b).correlation
    return float(value) if value is not None else np.nan
