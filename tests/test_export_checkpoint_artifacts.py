import numpy as np

from src.export_checkpoint_artifacts import summarize_pairwise


def test_summarize_pairwise_identical_matrices():
    matrix = np.array(
        [
            [0.1, 0.4, 0.2, 0.3],
            [0.7, 0.2, 0.5, 0.1],
        ]
    )
    summary = summarize_pairwise(
        "toy",
        "RF",
        [
            ("a", matrix),
            ("b", matrix.copy()),
            ("c", matrix.copy()),
        ],
    )

    assert summary["n_models"] == 3
    assert summary["n_pairs"] == 3
    assert summary["n_reference_samples"] == 2
    assert np.isclose(summary["spearman_mean"], 1.0)
    assert np.isclose(summary["jaccard5_mean"], 1.0)
    assert np.isclose(summary["jaccard10_mean"], 1.0)
