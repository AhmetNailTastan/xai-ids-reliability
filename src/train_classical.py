from __future__ import annotations

from sklearn.metrics import f1_score

from .data_preprocessing import iter_fold_safe_arrays


def evaluate_model_fold_safe(model_factory, X_raw, y, *, n_splits=5, seed=42) -> list[dict]:
    records = []
    for fold_data in iter_fold_safe_arrays(X_raw, y, n_splits=n_splits, seed=seed):
        model = model_factory(seed)
        model.fit(fold_data["X_train"], fold_data["y_train"])
        y_pred = model.predict(fold_data["X_test"])
        records.append(
            {
                "fold": fold_data["fold"],
                "seed": seed,
                "f1_macro": f1_score(fold_data["y_test"], y_pred, average="macro"),
                "model": model,
                "preprocessor": fold_data["preprocessor"],
            }
        )
    return records
