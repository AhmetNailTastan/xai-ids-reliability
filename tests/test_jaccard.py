import numpy as np

from src.model_variation_pairwise import compute_pairwise_variation, jaccard_top_k


def test_jaccard_top_k_identical_rankings():
    a = np.array([0.1, 0.2, 0.9, 0.7])
    b = np.array([0.1, 0.2, 0.9, 0.7])
    assert jaccard_top_k(a, b, 2) == 1.0


def test_pairwise_variation_counts_pairs():
    shap_items = [
        ("m1", np.array([[1, 2, 3], [1, 3, 2]], dtype=float)),
        ("m2", np.array([[1, 2, 4], [1, 4, 2]], dtype=float)),
        ("m3", np.array([[3, 2, 1], [2, 3, 1]], dtype=float)),
    ]
    records = compute_pairwise_variation("toy", shap_items, n_samples=2)
    assert len(records) == 3
    assert {record["model"] for record in records} == {"toy"}
