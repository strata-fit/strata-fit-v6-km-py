from __future__ import annotations

import argparse
import glob
import json

from io import StringIO
from pathlib import Path

import pandas as pd

from strata_fit_v6_km_py.central import run_local_kaplan_meier
from strata_fit_v6_km_py.types import DEFAULT_CUMULATIVE_INCIDENCE_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the STRATA-FIT KM flow locally against CSV files."
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Path to a CSV dataset. Repeat for multiple organizations.",
    )
    parser.add_argument(
        "--dataset-glob",
        help="Glob pattern for dataset files, e.g. 'tests/data/data_times/*.csv'.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render the cumulative-incidence plot with matplotlib.",
    )
    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.datasets]
    if args.dataset_glob:
        paths.extend(Path(value) for value in sorted(glob.glob(args.dataset_glob)))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    if len(unique_paths) < 3:
        raise ValueError("Provide at least three dataset files for the KM run")
    return unique_paths


def plot_km_curve(df_km: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    years = df_km["interval_start"] / 12
    cum_inc = df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN]

    plt.figure(figsize=(8, 5))
    plt.step(years, cum_inc, where="post", lw=2)
    plt.xlabel("Years from diagnosis")
    plt.ylabel("Cumulative incidence of D2T-RA")
    plt.title("Cumulative incidence of difficult-to-treat RA (KM estimate)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    dataset_paths = resolve_dataset_paths(args)
    datasets = [pd.read_csv(path) for path in dataset_paths]

    result = run_local_kaplan_meier(datasets)
    df_km = pd.read_json(StringIO(result["km_result"]))
    df_prev = pd.read_json(StringIO(result["d2t_prevalence"]))

    print(json.dumps(result["metadata"], indent=2))
    print("\nKaplan-Meier curve (first 5 rows):")
    print(df_km.head(), "\n")

    print("Summary statistics:")
    print(df_km.describe())

    print("\nD2T-RA prevalence by calendar year:")
    print(df_prev)

    assert not df_km.empty
    assert df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN].max() <= 1.0

    if args.plot:
        plot_km_curve(df_km)


if __name__ == "__main__":
    main()
