from __future__ import annotations

from .config import LIME_NUM_SAMPLES, LIME_RANDOM_STATE, N_MAIN_XAI, N_STOCHASTIC_INSTANCES, N_STOCHASTIC_REPEATS


def normalise_dataset_name(dataset: str) -> str:
    aliases = {
        "edge": "edgeiiot",
        "edge-iiot": "edgeiiot",
        "edgeiiot": "edgeiiot",
        "cic": "cicids2017",
        "cicids": "cicids2017",
        "cicids2017": "cicids2017",
        "cic-ids-2017": "cicids2017",
    }
    key = dataset.lower().replace("_", "-")
    if key not in aliases:
        raise ValueError(f"Unknown dataset: {dataset}")
    return aliases[key]


def main_checkpoint_name(dataset: str, model: str) -> str:
    return f"xai_{normalise_dataset_name(dataset)}_main{N_MAIN_XAI}_{model}.pkl"


def stochastic_checkpoint_name(dataset: str, model: str) -> str:
    return f"lime_stochastic_{normalise_dataset_name(dataset)}_{N_STOCHASTIC_INSTANCES}x{N_STOCHASTIC_REPEATS}_{model}.pkl"


def grid_checkpoint_name(dataset: str, model: str) -> str:
    return f"lime_grid_{normalise_dataset_name(dataset)}_{model}.pkl"


def lime_kwargs_for_main() -> dict:
    return {
        "num_samples": LIME_NUM_SAMPLES,
        "random_state": LIME_RANDOM_STATE,
    }


def lime_kwargs_for_stochastic(repeat_index: int) -> dict:
    return {
        "num_samples": LIME_NUM_SAMPLES,
        "random_state": None if repeat_index is None else repeat_index,
    }
