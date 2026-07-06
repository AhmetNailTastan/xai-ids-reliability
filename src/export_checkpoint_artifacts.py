from __future__ import annotations

import argparse
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata


CLASSICAL_MODEL_ORDER = [
    "RandomForest",
    "ExtraTrees",
    "DecisionTree",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "LogisticRegression",
]

MODEL_LABELS = {
    "RandomForest": "RF",
    "ExtraTrees": "ET",
    "DecisionTree": "DT",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CB",
    "LogisticRegression": "LR",
}

TREE_MODELS = {
    "RandomForest",
    "ExtraTrees",
    "DecisionTree",
    "XGBoost",
    "LightGBM",
    "CatBoost",
}


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_feature_names(feature_names, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"feature_index": range(len(feature_names)), "feature_name": list(feature_names)}
    ).to_csv(out, index=False)


def export_feature_metadata(edge_veri: str | Path, cic_veri: str | Path, metadata_dir: str | Path) -> None:
    metadata_dir = Path(metadata_dir)

    edge = load_pickle(edge_veri)
    cic = load_pickle(cic_veri)

    write_feature_names(edge["feature_names_multi"], metadata_dir / "edgeiiot_feature_names.csv")
    write_feature_names(cic["feature_names"], metadata_dir / "cicids2017_feature_names.csv")


def normalize_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        arr = np.asarray(values)
        arr = np.mean(np.abs(arr), axis=0)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = np.mean(np.abs(arr), axis=2)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP matrix after normalization, got {arr.shape}")

    return np.asarray(arr, dtype=float)


def get_shap_matrix(model_name: str, model, X_reference: np.ndarray) -> np.ndarray:
    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_reference)

    values = explainer.shap_values(X_reference)
    return normalize_shap_values(values)


def make_reference_sample(checkpoint_dir: str | Path, pattern: str, sample_size: int) -> np.ndarray:
    seed0 = load_pickle(Path(checkpoint_dir) / pattern.format(seed=0))
    ref_x_test = np.asarray(seed0[0]["X_test"])

    if len(ref_x_test) < sample_size:
        raise ValueError(f"Reference test set has {len(ref_x_test)} rows, cannot sample {sample_size}")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(ref_x_test), size=sample_size, replace=False)
    X_reference = ref_x_test[idx]

    del seed0
    gc.collect()
    return X_reference


def row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_rank = np.apply_along_axis(rankdata, 1, a)
    b_rank = np.apply_along_axis(rankdata, 1, b)

    a_rank = a_rank - a_rank.mean(axis=1, keepdims=True)
    b_rank = b_rank - b_rank.mean(axis=1, keepdims=True)

    numerator = np.sum(a_rank * b_rank, axis=1)
    denominator = np.sqrt(np.sum(a_rank * a_rank, axis=1) * np.sum(b_rank * b_rank, axis=1))

    out = np.full(a.shape[0], np.nan, dtype=float)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def row_jaccard_top_k(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    if k > a.shape[1]:
        k = a.shape[1]

    top_a = np.argpartition(np.abs(a), -k, axis=1)[:, -k:]
    top_b = np.argpartition(np.abs(b), -k, axis=1)[:, -k:]

    scores = np.empty(a.shape[0], dtype=float)
    for i, (idx_a, idx_b) in enumerate(zip(top_a, top_b, strict=True)):
        set_a = set(idx_a.tolist())
        set_b = set(idx_b.tolist())
        scores[i] = len(set_a & set_b) / len(set_a | set_b)
    return scores


def prepare_pairwise_items(shap_items: list[tuple[str, np.ndarray]]) -> list[dict]:
    prepared = []
    for item_id, matrix in shap_items:
        matrix = np.asarray(matrix, dtype=float)
        ranks = np.apply_along_axis(rankdata, 1, matrix)
        ranks = ranks - ranks.mean(axis=1, keepdims=True)
        rank_norm = np.sqrt(np.sum(ranks * ranks, axis=1))

        prepared.append(
            {
                "id": item_id,
                "matrix": matrix,
                "rank_centered": ranks,
                "rank_norm": rank_norm,
                "top5": top_k_indices(matrix, 5),
                "top10": top_k_indices(matrix, 10),
            }
        )
    return prepared


def top_k_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    if k > matrix.shape[1]:
        k = matrix.shape[1]
    return np.argpartition(np.abs(matrix), -k, axis=1)[:, -k:]


def prepared_spearman(a: dict, b: dict, n_samples: int) -> np.ndarray:
    numerator = np.sum(
        a["rank_centered"][:n_samples] * b["rank_centered"][:n_samples],
        axis=1,
    )
    denominator = a["rank_norm"][:n_samples] * b["rank_norm"][:n_samples]

    out = np.full(n_samples, np.nan, dtype=float)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def prepared_jaccard(a_top: np.ndarray, b_top: np.ndarray, n_samples: int) -> np.ndarray:
    scores = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        set_a = set(a_top[i].tolist())
        set_b = set(b_top[i].tolist())
        scores[i] = len(set_a & set_b) / len(set_a | set_b)
    return scores


def summarize_pairwise(dataset: str, model_label: str, shap_items: list[tuple[str, np.ndarray]]) -> dict:
    prepared_items = prepare_pairwise_items(shap_items)
    n_models = len(prepared_items)
    n_reference_samples = min(item["matrix"].shape[0] for item in prepared_items)

    pair_metrics = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a = prepared_items[i]
            b = prepared_items[j]
            pair_metrics.append(
                (
                    np.nanmean(prepared_spearman(a, b, n_reference_samples)),
                    np.nanmean(prepared_jaccard(a["top5"], b["top5"], n_reference_samples)),
                    np.nanmean(prepared_jaccard(a["top10"], b["top10"], n_reference_samples)),
                )
            )

    metrics = np.asarray(pair_metrics, dtype=float)
    return {
        "dataset": dataset,
        "model": model_label,
        "n_models": n_models,
        "n_pairs": len(pair_metrics),
        "n_reference_samples": n_reference_samples,
        "spearman_mean": np.nanmean(metrics[:, 0]),
        "spearman_std": np.nanstd(metrics[:, 0], ddof=1),
        "jaccard5_mean": np.nanmean(metrics[:, 1]),
        "jaccard5_std": np.nanstd(metrics[:, 1], ddof=1),
        "jaccard10_mean": np.nanmean(metrics[:, 2]),
        "jaccard10_std": np.nanstd(metrics[:, 2], ddof=1),
    }


def compute_classical_pairwise(
    dataset: str,
    checkpoint_dir: str | Path,
    pattern: str,
    *,
    sample_size: int,
    seeds: int,
) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    X_reference = make_reference_sample(checkpoint_dir, pattern, sample_size)
    summaries = []

    for model_name in CLASSICAL_MODEL_ORDER:
        started = time.time()
        shap_items: list[tuple[str, np.ndarray]] = []
        print(f"[{dataset}] {model_name}: computing SHAP matrices", flush=True)

        for seed in range(seeds):
            seed_data = load_pickle(checkpoint_dir / pattern.format(seed=seed))
            for record in seed_data:
                if record["model_adi"] != model_name:
                    continue

                item_id = f"seed{record['seed']}_fold{record['fold']}"
                matrix = get_shap_matrix(model_name, record["model_obj"], X_reference)
                shap_items.append((item_id, matrix))

            del seed_data
            gc.collect()

        summary = summarize_pairwise(dataset, MODEL_LABELS[model_name], shap_items)
        summary["status"] = "true_pairwise_from_model_checkpoints"
        summary["source"] = f"model_checkpoints:{checkpoint_dir.name}"
        summaries.append(summary)

        elapsed = time.time() - started
        print(f"[{dataset}] {model_name}: {summary['n_pairs']} pairs in {elapsed:.1f}s", flush=True)

        del shap_items
        gc.collect()

    return summaries


def compute_neural_pairwise(dataset: str, model_label: str, path: str | Path) -> dict:
    data = load_pickle(path)
    shap_items = [(f"seed{k[0]}_fold{k[1]}", np.asarray(v, dtype=float)) for k, v in sorted(data.items())]
    summary = summarize_pairwise(dataset, model_label, shap_items)
    summary["status"] = "true_pairwise_from_saved_shap_arrays"
    summary["source"] = f"saved_shap_arrays:{Path(path).name}"
    return summary


def write_table_08(rows: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "model",
        "n_models",
        "n_pairs",
        "n_reference_samples",
        "spearman_mean",
        "spearman_std",
        "jaccard5_mean",
        "jaccard5_std",
        "jaccard10_mean",
        "jaccard10_std",
        "status",
        "source",
    ]
    df = pd.DataFrame(rows, columns=columns)
    for col in [
        "spearman_mean",
        "spearman_std",
        "jaccard5_mean",
        "jaccard5_std",
        "jaccard10_mean",
        "jaccard10_std",
    ]:
        df[col] = df[col].astype(float).round(6)
    df.to_csv(out, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export metadata and true pairwise RQ2 summaries from local checkpoints."
    )
    parser.add_argument("--edge-checkpoints", default="data/edgeiiot_checkpoints")
    parser.add_argument("--cic-checkpoints", default="data/cicids2017_checkpoints")
    parser.add_argument("--dnn-lstm-checkpoints", default="data/dnn_lstm_checkpoints")
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--out", default="results/table_08_rq2_model_variation_pairwise.csv")
    parser.add_argument("--classical-sample-size", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--skip-classical-rq2", action="store_true")
    parser.add_argument("--skip-neural-rq2", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    edge_dir = Path(args.edge_checkpoints)
    cic_dir = Path(args.cic_checkpoints)
    dnn_dir = Path(args.dnn_lstm_checkpoints)

    if not args.skip_metadata:
        export_feature_metadata(edge_dir / "veri.pkl", cic_dir / "veri_cicids.pkl", args.metadata_dir)

    rows: list[dict] = []
    if not args.skip_classical_rq2:
        rows.extend(
            compute_classical_pairwise(
                "Edge-IIoT",
                edge_dir,
                "modeller_seed{seed}.pkl",
                sample_size=args.classical_sample_size,
                seeds=args.seeds,
            )
        )
        rows.extend(
            compute_classical_pairwise(
                "CIC-IDS-2017",
                cic_dir,
                "modeller_cicids_seed{seed}.pkl",
                sample_size=args.classical_sample_size,
                seeds=args.seeds,
            )
        )

    if not args.skip_neural_rq2:
        rows.extend(
            [
                compute_neural_pairwise("Edge-IIoT", "DNN", dnn_dir / "rq2_dnn_edge.pkl"),
                compute_neural_pairwise("CIC-IDS-2017", "DNN", dnn_dir / "rq2_dnn_cic.pkl"),
                compute_neural_pairwise("Edge-IIoT", "LSTM", dnn_dir / "rq2_lstm_edge.pkl"),
                compute_neural_pairwise("CIC-IDS-2017", "LSTM", dnn_dir / "rq2_lstm_cic.pkl"),
            ]
        )

    if rows:
        order = {
            "RF": 0,
            "ET": 1,
            "DT": 2,
            "XGB": 3,
            "LGBM": 4,
            "CB": 5,
            "LR": 6,
            "DNN": 7,
            "LSTM": 8,
        }
        rows.sort(key=lambda row: (order[row["model"]], row["dataset"]))
        write_table_08(rows, args.out)


if __name__ == "__main__":
    main()
