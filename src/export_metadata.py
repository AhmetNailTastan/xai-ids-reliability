from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_feature_names(columns, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"feature_index": range(len(columns)), "feature_name": list(columns)}
    ).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight feature metadata from a prepared CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default=None, help="Optional target column to exclude.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=1)
    columns = [col for col in df.columns if col != args.target]
    write_feature_names(columns, args.out)


if __name__ == "__main__":
    main()
