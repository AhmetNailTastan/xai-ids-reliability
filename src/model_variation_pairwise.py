from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def jaccard_top_k(a, b, k):
    top_a = set(np.argsort(np.abs(a))[-k:])
    top_b = set(np.argsort(np.abs(b))[-k:])
    return len(top_a & top_b) / len(top_a | top_b)


def compute_pairwise_variation(model_name, shap_items, n_samples=500):
    """
    shap_items: list of tuples [(instance_id, shap_matrix), ...]
    shap_matrix shape: (n_samples, n_features)
    """
    records = []
    for (id_a, shap_a), (id_b, shap_b) in combinations(shap_items, 2):
        sp_values, j5_values, j10_values = [], [], []

        for i in range(n_samples):
            sp_values.append(spearmanr(shap_a[i], shap_b[i]).correlation)
            j5_values.append(jaccard_top_k(shap_a[i], shap_b[i], 5))
            j10_values.append(jaccard_top_k(shap_a[i], shap_b[i], 10))

        records.append(
            {
                "model": model_name,
                "pair_a": id_a,
                "pair_b": id_b,
                "spearman": np.nanmean(sp_values),
                "jaccard_5": np.nanmean(j5_values),
                "jaccard_10": np.nanmean(j10_values),
            }
        )

    return records


def summarize_pairwise_records(dataset: str, model_name: str, records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    return {
        "dataset": dataset,
        "model": model_name,
        "n_models": infer_n_models(records),
        "n_pairs": len(records),
        "spearman_mean": df["spearman"].mean(),
        "spearman_std": df["spearman"].std(ddof=1),
        "jaccard5_mean": df["jaccard_5"].mean(),
        "jaccard5_std": df["jaccard_5"].std(ddof=1),
        "jaccard10_mean": df["jaccard_10"].mean(),
        "jaccard10_std": df["jaccard_10"].std(ddof=1),
    }


def infer_n_models(records: list[dict]) -> int:
    ids = set()
    for record in records:
        ids.add(record["pair_a"])
        ids.add(record["pair_b"])
    return len(ids)


def load_shap_items_from_npz(path: str | Path) -> list[tuple[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    return [(key, data[key]) for key in sorted(data.files)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute true pairwise RQ2 model-variation consistency.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--npz", required=True, help="NPZ with one SHAP matrix per trained model instance.")
    parser.add_argument("--out", default="results/table_08_rq2_model_variation_pairwise.csv")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    shap_items = load_shap_items_from_npz(args.npz)
    records = compute_pairwise_variation(args.model, shap_items, n_samples=args.n_samples)
    summary = summarize_pairwise_records(args.dataset, args.model, records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame([summary])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
