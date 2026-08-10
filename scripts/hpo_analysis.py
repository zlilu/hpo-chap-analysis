#!/usr/bin/env python3

from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import get_metrics_registry


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
CHAP_CORE_ROOT = ANALYSIS_ROOT.parent / "chap-core"

# MODEL_NAME = "https://github.com/chap-models/minimal_template_example"
# MODEL_NAME = "https://github.com/chap-models/chtorch.git"
MODEL_NAME = "https://github.com/chap-models/auto_regressive_monthly_v2"
DATASET_CSV = CHAP_CORE_ROOT / "example_data/vietnam_monthly.csv"
CONFIG_YAML = ANALYSIS_ROOT / "scripts/config.yaml"
OUTPUT_DIR = ANALYSIS_ROOT / "results/simple_hpo_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERIODS = 3
N_SPLITS = 12 # monthly
STRIDE = 1
HPO_METRIC = "rmse"
# N_TRIALS = 100
# SEARCHERS = ["random", "tpe"]


def run_command(command, log_file):
    """Run one shell command and save stdout/stderr to a log file."""
    print("\nRunning:")
    print(" ".join(command))

    start = time.time()

    with open(log_file, "w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    seconds = time.time() - start

    print(f"Finished with exit code {result.returncode} in {seconds:.1f} seconds")
    print(f"Log written to: {log_file}")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed. See log: {log_file}")

    return seconds


def make_base_command(output_file):
    """Create the shared part of the chap eval command."""
    return [
        "chap",
        "eval",
        "--model-name",
        MODEL_NAME,
        "--dataset-csv",
        str(DATASET_CSV),
        "--output-file",
        str(output_file),
        # "--model-configuration-yaml",
        # CONFIG_YAML,
        "--backtest-params.n-periods",
        str(N_PERIODS),
        "--backtest-params.n-splits",
        str(N_SPLITS),
        "--backtest-params.stride",
        str(STRIDE),
    ]


def compute_all_metrics(evaluation):
    """
    Compute every registered metric that is applicable to this evaluation.

    Returns:
        dict[str, float]
    """
    flat_data = evaluation.to_flat()

    historical_df = None
    if flat_data.historical_observations is not None:
        historical_df = pd.DataFrame(
            flat_data.historical_observations
        )

    results = {}

    for metric_id, metric_cls in get_metrics_registry().items():
        try:
            metric = metric_cls(
                historical_observations=historical_df
            )

            if not metric.is_applicable(flat_data.observations):
                print(f"Skipping {metric_id}: not applicable")
                continue

            metric_df = metric.get_global_metric(
                flat_data.observations,
                flat_data.forecasts,
            )

            if len(metric_df) != 1:
                print(
                    f"Skipping {metric_id}: "
                    f"expected 1 result, got {len(metric_df)}"
                )
                continue

            results[metric_id] = float(
                metric_df["metric"].iloc[0]
            )

        except Exception as exc:
            print(f"Skipping {metric_id}: {exc}")

    return results


def compare_metrics(fixed_metrics, hpo_metrics):
    """Create a simple fixed-vs-HPO comparison table."""
    metric_ids = sorted(
        set(fixed_metrics) | set(hpo_metrics)
    )

    rows = []

    for metric_id in metric_ids:
        fixed = fixed_metrics.get(metric_id)
        hpo = hpo_metrics.get(metric_id)

        delta = None
        relative_change = None

        if fixed is not None and hpo is not None:
            delta = hpo - fixed

            if fixed != 0:
                relative_change = (
                    delta / abs(fixed)
                ) * 100

        rows.append(
            {
                "metric": metric_id,
                "fixed": fixed,
                "hpo": hpo,
                "delta_hpo_minus_fixed": delta,
                "relative_change_pct": relative_change,
                "hpo_objective": metric_id == HPO_METRIC,
            }
        )

    return pd.DataFrame(rows)


def inspect_netcdf(nc_file):
    """Print basic information about the NetCDF output."""
    print(f"\nInspecting: {nc_file}")

    with xr.open_dataset(nc_file) as ds:
        print("\nDataset dimensions:")
        print(dict(ds.sizes))

        print("\nDataset variables:")
        for name in ds.data_vars:
            variable = ds[name]
            print(f"- {name}: dims={variable.dims}, shape={variable.shape}, dtype={variable.dtype}")

        print("\nDataset attributes:")
        for key, value in ds.attrs.items():
            print(f"- {key}: {value}")


def main():
    normal_output = OUTPUT_DIR / "normal_eval.nc"
    hpo_output = OUTPUT_DIR / "hpo_eval.nc"

    normal_log = OUTPUT_DIR / "normal_eval.log"
    hpo_log = OUTPUT_DIR / "hpo_eval.log"

    normal_command = make_base_command(normal_output)

    hpo_command = make_base_command(hpo_output) + [
        "--model-configuration-yaml",
        str(CONFIG_YAML),
        "--estimator-options.mode",
        "hpo",
        "--estimator-options.metric",
        HPO_METRIC,
    ]

    normal_seconds = run_command(normal_command, normal_log)
    hpo_seconds = run_command(hpo_command, hpo_log)

    normal_evaluation = Evaluation.from_file(normal_output)
    hpo_evaluation = Evaluation.from_file(hpo_output)

    print("\nComputing fixed-model metrics...")
    normal_metrics = compute_all_metrics(normal_evaluation)

    print("\nComputing HPO-model metrics...")
    hpo_metrics = compute_all_metrics(hpo_evaluation)

    comparison = compare_metrics(
        normal_metrics,
        hpo_metrics,
    )

    output_csv = OUTPUT_DIR / "metric_comparison.csv"
    comparison.to_csv(output_csv, index=False)

    print("\nNormal vs HPO:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nRuntime comparison:")
    print(f"- normal: {normal_seconds:.1f} seconds")
    print(f"- hpo:    {hpo_seconds:.1f} seconds")
    print(f"HPO / fixed: {hpo_seconds / normal_seconds:.2f}x")

    print(f"\nResults written to:")
    print(output_csv)

    inspect_netcdf(normal_output)
    inspect_netcdf(hpo_output)


if __name__ == "__main__":
    main()