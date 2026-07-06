from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_types(
    X: pd.DataFrame,
    categorical_cols: Sequence[str] | None = None,
    numeric_cols: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Column inference requires a pandas DataFrame")

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        categorical_cols = list(categorical_cols)

    if numeric_cols is None:
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
    else:
        numeric_cols = list(numeric_cols)

    return numeric_cols, categorical_cols


def make_fold_safe_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] | None = None,
) -> ColumnTransformer:
    categorical_cols = list(categorical_cols or [])
    transformers = []

    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), list(numeric_cols)))

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("At least one numeric or categorical column is required")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def iter_fold_safe_arrays(
    X_raw,
    y,
    *,
    n_splits: int = 5,
    seed: int = 42,
    numeric_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> Iterator[dict]:
    """Yield train/test arrays with preprocessing fitted inside each fold only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if isinstance(X_raw, pd.DataFrame):
        numeric_cols, categorical_cols = infer_column_types(X_raw, categorical_cols, numeric_cols)
        indexer = X_raw.iloc
    else:
        X_raw = np.asarray(X_raw)
        numeric_cols = list(range(X_raw.shape[1]))
        categorical_cols = []
        indexer = None

    y = np.asarray(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), start=1):
        if indexer is not None:
            X_train_raw = X_raw.iloc[train_idx]
            X_test_raw = X_raw.iloc[test_idx]
        else:
            X_train_raw = X_raw[train_idx]
            X_test_raw = X_raw[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        preprocessor = make_fold_safe_preprocessor(numeric_cols, categorical_cols)

        # Fold-safe preprocessing: scaler is fitted only on training data.
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        yield {
            "fold": fold,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor": preprocessor,
        }
