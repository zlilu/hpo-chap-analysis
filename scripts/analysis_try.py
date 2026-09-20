#!/usr/bin/env python3
"""Comprehensive CHAP HPO experiment runner and analysis.

This script exercises the local ``chap eval`` CLI in both normal and HPO modes
for two model templates and three monthly climate-health datasets.  It is aimed
at a thesis / master's-level empirical assessment of CHAP's HPO implementation,
including:

- normal-vs-HPO forecast performance on multiple metrics;
- RandomSearcher-vs-TPESearcher comparisons;
- 20/50/100-trial budget experiments;
- repeated-seed analysis;
- trial-order reconstruction from the score-sorted leaderboard using trial_nr;
- best-so-far convergence by trial and cumulative objective time;
- time-to-best and time-to-95%-of-improvement;
- run-time scaling and HPO overhead;
- failed-trial rates, failure classes, and failure hot-spots in the search space;
- duplicate-configuration rates;
- internal-HPO-score vs outer-evaluation score (selection/generalization gap);
- cross-budget prefix consistency as a reproducibility diagnostic;
- best-hyperparameter stability across repeated runs;
- horizon-wise and location-wise evaluation heterogeneity;
- exploratory paired TPE-vs-random statistics and bootstrap confidence intervals.

The script intentionally treats ``chap eval`` as the system-under-test rather
than instantiating HyperparameterOptimizer directly.  That keeps the experiment
aligned with the user-facing CLI integration path.

Expected repository layout (matching the original analysis script):

    <workspace>/
      chap-core/
      <analysis-repo>/
        scripts/hpo_master_analysis.py
        scripts/auto_reg_monthly_v2_conf.yaml
        scripts/minimal_template_config_hpo.yaml

The two YAML files above are HPO search-space definitions and are passed via
``--estimator-options.search-space``. They are not model configuration files.

The local CHAP checkout is preferred via ``uv run --project <chap-core> chap``.
Set CHAP_COMMAND or pass --chap-command to override this.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Resolve local chap-core before importing chap_core.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
CHAP_CORE_ROOT = Path(
    os.environ.get("CHAP_CORE_ROOT", str(ANALYSIS_ROOT.parent / "chap-core"))
).expanduser().resolve()

if CHAP_CORE_ROOT.exists():
    sys.path.insert(0, str(CHAP_CORE_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import get_metrics_registry, get_optimization_direction


SCRIPT_VERSION = "2.2"
REQUESTED_METRICS = ("crps_log1p", "crps", "mae", "rmse")
DEFAULT_BUDGETS = (1, 2) # (20, 50, 100)
# Three repetitions are a pragmatic default for stochastic-search variability.
# For formal thesis inference, increase to >=5 if compute permits.
DEFAULT_SEEDS = (17, 53) # (17, 53, 101)
DEFAULT_SEARCHERS = ("random", "tpe")
N_PERIODS = 3
N_SPLITS = 12
STRIDE = 1


@dataclass(frozen=True)
class ModelSpec:
    key: str
    url: str
    objective_metric: str
    configuration_yaml: Path | None = None
    search_space_yaml: Path | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    url: str


@dataclass(frozen=True)
class RunSpec:
    model: ModelSpec
    dataset: DatasetSpec
    mode: str  # normal | hpo
    searcher: str | None = None
    budget: int | None = None
    seed: int | None = None

    @property
    def run_id(self) -> str:
        parts = [self.model.key, self.dataset.key, self.mode]
        if self.mode == "hpo":
            parts.extend([str(self.searcher), f"b{self.budget}", f"s{self.seed}"])
        return "__".join(parts)


@dataclass
class CommandResult:
    returncode: int
    seconds: float
    skipped: bool
    command: list[str]


AUTO_REG_DEFAULT_SEARCH_SPACE = ANALYSIS_ROOT / "scripts" / "auto_reg_monthly_v2_conf.yaml"
MINIMAL_DEFAULT_SEARCH_SPACE = ANALYSIS_ROOT / "scripts" / "minimal_template_config_hpo.yaml"

BASE_MODEL_SPECS = {
    "auto_regressive_monthly_v2": ModelSpec(
        key="auto_regressive_monthly_v2",
        url="https://github.com/chap-models/auto_regressive_monthly_v2",
        objective_metric="crps_log1p",
        configuration_yaml=None,
        search_space_yaml=AUTO_REG_DEFAULT_SEARCH_SPACE,
    ),
    "minimal_template_example": ModelSpec(
        key="minimal_template_example",
        url="https://github.com/chap-models/minimal_template_example",
        objective_metric="rmse",
        configuration_yaml=None,
        search_space_yaml=MINIMAL_DEFAULT_SEARCH_SPACE,
    ),
}

DATASET_SPECS = {
    "lao": DatasetSpec(
        key="lao",
        url="https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv",
    ),
    "tha": DatasetSpec(
        key="tha",
        url="https://raw.githubusercontent.com/dhis2/climate-health-data/main/tha/chap_THA_admin1_monthly.csv",
    ),
    "vnm": DatasetSpec(
        key="vnm",
        url="https://raw.githubusercontent.com/dhis2/climate-health-data/main/vnm/chap_VNM_admin1_monthly.csv",
    ),
}


# ---------------------------------------------------------------------------
# CLI and filesystem helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def resolve_chap_command(user_value: str | None) -> list[str]:
    """Resolve a command that executes the local CHAP checkout when possible."""
    raw = user_value or os.environ.get("CHAP_COMMAND")
    if raw:
        return shlex.split(raw)

    if (CHAP_CORE_ROOT / "pyproject.toml").exists() and shutil.which("uv"):
        return ["uv", "run", "--project", str(CHAP_CORE_ROOT), "chap"]

    chap = shutil.which("chap")
    if chap:
        return [chap]

    raise RuntimeError(
        "Could not find a CHAP CLI. Install/activate the local chap-core environment, "
        "or pass --chap-command (for example: --chap-command 'uv run chap')."
    )


def run_directory(root: Path, spec: RunSpec) -> Path:
    if spec.mode == "normal":
        return root / "raw" / spec.model.key / spec.dataset.key / "normal"
    assert spec.searcher is not None and spec.budget is not None and spec.seed is not None
    return (
        root
        / "raw"
        / spec.model.key
        / spec.dataset.key
        / "hpo"
        / spec.searcher
        / f"budget_{spec.budget:03d}"
        / f"seed_{spec.seed}"
    )


def output_paths(root: Path, spec: RunSpec) -> dict[str, Path]:
    directory = run_directory(root, spec)
    output_file = directory / "evaluation.nc"
    return {
        "dir": directory,
        "nc": output_file,
        "log": directory / "evaluation.log",
        "command": directory / "command.txt",
        "metadata": directory / "run_metadata.json",
        "leaderboard": output_file.with_suffix(".hpo-leaderboard.csv"),
    }


def build_base_command(
    chap_command: Sequence[str],
    spec: RunSpec,
    output_file: Path,
    *,
    ignore_environment: bool,
) -> list[str]:
    cmd = [
        *chap_command,
        "eval",
        "--model-name",
        spec.model.url,
        "--dataset-csv",
        spec.dataset.url,
        "--output-file",
        str(output_file),
        "--backtest-params.n-periods",
        str(N_PERIODS),
        "--backtest-params.n-splits",
        str(N_SPLITS),
        "--backtest-params.stride",
        str(STRIDE),
    ]

    if spec.model.configuration_yaml is not None:
        cmd.extend(
            ["--model-configuration-yaml", str(spec.model.configuration_yaml)]
        )

    if ignore_environment:
        cmd.append("--run-config.ignore-environment")

    return cmd


def build_command(
    chap_command: Sequence[str],
    spec: RunSpec,
    output_file: Path,
    *,
    ignore_environment: bool,
) -> list[str]:
    cmd = build_base_command(
        chap_command,
        spec,
        output_file,
        ignore_environment=ignore_environment,
    )

    if spec.mode == "normal":
        return cmd

    if spec.mode != "hpo":
        raise ValueError(f"Unsupported mode: {spec.mode}")

    assert spec.searcher is not None
    assert spec.budget is not None
    assert spec.seed is not None

    # HPO experiments deliberately use an explicit search-space file.  Do not
    # fall back to model_template_config.hpo_search_space: the search space is
    # part of the experimental treatment and must therefore be versionable and
    # visible in the exact CLI command used for every run.
    if spec.model.search_space_yaml is None:
        raise ValueError(
            f"Explicit HPO search-space YAML is required for {spec.model.key}. "
            "Pass the corresponding --*-search-space argument."
        )

    cmd.extend(
        [
            "--estimator-options.mode",
            "hpo",
            "--estimator-options.search-space",
            str(spec.model.search_space_yaml),
            "--estimator-options.metric",
            spec.model.objective_metric,
            "--estimator-options.searcher",
            spec.searcher,
            "--estimator-options.max-trials",
            str(spec.budget),
            "--estimator-options.seed",
            str(spec.seed),
        ]
    )

    return cmd


def completed_run_exists(paths: dict[str, Path], mode: str) -> bool:
    if not paths["nc"].exists():
        return False
    if mode == "hpo" and not paths["leaderboard"].exists():
        return False
    return True


def execute_run(
    *,
    root: Path,
    spec: RunSpec,
    chap_command: Sequence[str],
    force: bool,
    ignore_environment: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    paths = output_paths(root, spec)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    command = build_command(
        chap_command,
        spec,
        paths["nc"],
        ignore_environment=ignore_environment,
    )
    paths["command"].write_text(shlex.join(command) + "\n", encoding="utf-8")

    if not force and completed_run_exists(paths, spec.mode):
        metadata = load_json(paths["metadata"]) or {}
        return {
            "run_id": spec.run_id,
            "model": spec.model.key,
            "dataset": spec.dataset.key,
            "mode": spec.mode,
            "searcher": spec.searcher,
            "budget": spec.budget,
            "seed": spec.seed,
            "objective_metric": spec.model.objective_metric,
            "search_space_yaml": str(spec.model.search_space_yaml) if spec.mode == "hpo" else None,
            "status": "cached",
            "returncode": int(metadata.get("returncode", 0)),
            "cli_seconds": safe_float(metadata.get("seconds")),
            "output_file": str(paths["nc"]),
            "leaderboard_file": (
                str(paths["leaderboard"]) if spec.mode == "hpo" else None
            ),
            "log_file": str(paths["log"]),
            "command": shlex.join(command),
        }

    print(f"\n[{utc_now()}] RUN {spec.run_id}")
    print(shlex.join(command))

    start = time.perf_counter()
    with paths["log"].open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=str(CHAP_CORE_ROOT) if CHAP_CORE_ROOT.exists() else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = time.perf_counter() - start

    metadata = {
        "run_id": spec.run_id,
        "model": spec.model.key,
        "model_url": spec.model.url,
        "dataset": spec.dataset.key,
        "dataset_url": spec.dataset.url,
        "mode": spec.mode,
        "searcher": spec.searcher,
        "budget": spec.budget,
        "seed": spec.seed,
        "objective_metric": spec.model.objective_metric,
        "search_space_yaml": str(spec.model.search_space_yaml) if spec.mode == "hpo" else None,
        "returncode": result.returncode,
        "seconds": seconds,
        "command": command,
        "completed_utc": utc_now(),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    status = "ok" if result.returncode == 0 else "failed"
    print(
        f"[{utc_now()}] {status.upper()} {spec.run_id}: "
        f"exit={result.returncode}, {seconds:.1f}s, log={paths['log']}"
    )

    if result.returncode != 0 and fail_fast:
        raise RuntimeError(f"CHAP command failed; see {paths['log']}")

    return {
        "run_id": spec.run_id,
        "model": spec.model.key,
        "dataset": spec.dataset.key,
        "mode": spec.mode,
        "searcher": spec.searcher,
        "budget": spec.budget,
        "seed": spec.seed,
        "objective_metric": spec.model.objective_metric,
        "search_space_yaml": str(spec.model.search_space_yaml) if spec.mode == "hpo" else None,
        "status": status,
        "returncode": result.returncode,
        "cli_seconds": seconds,
        "output_file": str(paths["nc"]),
        "leaderboard_file": (
            str(paths["leaderboard"]) if spec.mode == "hpo" else None
        ),
        "log_file": str(paths["log"]),
        "command": shlex.join(command),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Experiment construction and provenance
# ---------------------------------------------------------------------------


def build_run_specs(
    models: Sequence[ModelSpec],
    datasets: Sequence[DatasetSpec],
    searchers: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for model in models:
        for dataset in datasets:
            runs.append(RunSpec(model=model, dataset=dataset, mode="normal"))
            for searcher in searchers:
                for budget in budgets:
                    for seed in seeds:
                        runs.append(
                            RunSpec(
                                model=model,
                                dataset=dataset,
                                mode="hpo",
                                searcher=searcher,
                                budget=int(budget),
                                seed=int(seed),
                            )
                        )
    return runs


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(args: Sequence[str]) -> str | None:
    if not CHAP_CORE_ROOT.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(CHAP_CORE_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_provenance(
    root: Path,
    *,
    chap_command: Sequence[str],
    models: Sequence[ModelSpec],
    datasets: Sequence[DatasetSpec],
    searchers: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> None:
    provenance = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "chap_command": list(chap_command),
        "chap_core_root": str(CHAP_CORE_ROOT),
        "chap_git_commit": git_value(["rev-parse", "HEAD"]),
        "chap_git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "chap_git_status": git_value(["status", "--short"]),
        "packages": {
            "chap-core": package_version("chap-core"),
            "optuna": package_version("optuna"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "xarray": package_version("xarray"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
        "design": {
            "models": [asdict(m) for m in models],
            "datasets": [asdict(d) for d in datasets],
            "searchers": list(searchers),
            "budgets": list(map(int, budgets)),
            "seeds": list(map(int, seeds)),
            "metrics": list(REQUESTED_METRICS),
            "backtest": {
                "n_periods": N_PERIODS,
                "n_splits": N_SPLITS,
                "stride": STRIDE,
            },
        },
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------


def profile_dataset(spec: DatasetSpec) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": spec.key, "url": spec.url}
    try:
        df = pd.read_csv(spec.url)
    except Exception as exc:
        row["profile_error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["n_rows"] = len(df)
    row["n_columns"] = len(df.columns)
    row["columns"] = json_dumps(list(df.columns))

    if "location" in df:
        row["n_locations"] = int(df["location"].nunique(dropna=True))
    if "time_period" in df:
        periods = df["time_period"].astype(str)
        row["first_period"] = periods.min() if len(periods) else None
        row["last_period"] = periods.max() if len(periods) else None
        row["n_unique_periods"] = int(periods.nunique(dropna=True))

    if "disease_cases" in df:
        y = pd.to_numeric(df["disease_cases"], errors="coerce")
        row["disease_missing_pct"] = float(y.isna().mean() * 100)
        valid = y.dropna()
        if len(valid):
            row["disease_zero_pct"] = float((valid == 0).mean() * 100)
            row["disease_mean"] = float(valid.mean())
            row["disease_std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            row["disease_median"] = float(valid.median())
            row["disease_q95"] = float(valid.quantile(0.95))
            row["disease_max"] = float(valid.max())

    protected = {"time_period", "location", "location_name", "disease_cases"}
    candidate_covariates = [c for c in df.columns if c not in protected]
    row["candidate_covariates"] = json_dumps(candidate_covariates)
    row["covariate_missing_pct"] = json_dumps(
        {
            c: round(float(df[c].isna().mean() * 100), 4)
            for c in candidate_covariates
        }
    )
    return row


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def historical_dataframe(evaluation: Evaluation) -> pd.DataFrame | None:
    flat = evaluation.to_flat()
    if flat.historical_observations is None:
        return None
    return pd.DataFrame(flat.historical_observations)


def calculate_metric_on_frames(
    metric_id: str,
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
    historical_df: pd.DataFrame | None,
) -> tuple[float | None, str | None]:
    registry = get_metrics_registry()
    metric_cls = registry.get(metric_id)
    if metric_cls is None:
        return None, f"metric_not_registered:{metric_id}"

    try:
        metric = metric_cls(historical_observations=historical_df)
        if not metric.is_applicable(observations):
            return None, "not_applicable"
        metric_df = metric.get_global_metric(observations, forecasts)
        if len(metric_df) != 1:
            return None, f"unexpected_result_rows:{len(metric_df)}"
        value = safe_float(metric_df["metric"].iloc[0])
        return value, None if value is not None else "non_finite_metric"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compute_evaluation_metrics(
    nc_path: Path,
    *,
    run_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = Evaluation.from_file(nc_path)
    flat = evaluation.to_flat()
    observations = pd.DataFrame(flat.observations).copy()
    forecasts = pd.DataFrame(flat.forecasts).copy()
    historical_df = historical_dataframe(evaluation)

    base = {
        "run_id": run_row["run_id"],
        "model": run_row["model"],
        "dataset": run_row["dataset"],
        "mode": run_row["mode"],
        "searcher": run_row.get("searcher"),
        "budget": run_row.get("budget"),
        "seed": run_row.get("seed"),
        "objective_metric": run_row["objective_metric"],
    }

    global_rows: list[dict[str, Any]] = []
    for metric_id in REQUESTED_METRICS:
        value, error = calculate_metric_on_frames(
            metric_id, observations, forecasts, historical_df
        )
        global_rows.append(
            {
                **base,
                "metric": metric_id,
                "value": value,
                "metric_error": error,
                "is_objective": metric_id == run_row["objective_metric"],
            }
        )

    horizon_rows: list[dict[str, Any]] = []
    if "horizon_distance" in forecasts.columns:
        for horizon in sorted(forecasts["horizon_distance"].dropna().unique()):
            f_subset = forecasts.loc[forecasts["horizon_distance"] == horizon]
            for metric_id in REQUESTED_METRICS:
                value, error = calculate_metric_on_frames(
                    metric_id, observations, f_subset, historical_df
                )
                horizon_rows.append(
                    {
                        **base,
                        "horizon_distance": int(horizon),
                        "metric": metric_id,
                        "value": value,
                        "metric_error": error,
                    }
                )

    location_rows: list[dict[str, Any]] = []
    if "location" in forecasts.columns and "location" in observations.columns:
        locations = sorted(set(forecasts["location"].dropna().astype(str)))
        for location in locations:
            f_subset = forecasts.loc[forecasts["location"].astype(str) == location]
            o_subset = observations.loc[observations["location"].astype(str) == location]
            h_subset = historical_df
            if historical_df is not None and "location" in historical_df.columns:
                h_subset = historical_df.loc[
                    historical_df["location"].astype(str) == location
                ]
            for metric_id in REQUESTED_METRICS:
                value, error = calculate_metric_on_frames(
                    metric_id, o_subset, f_subset, h_subset
                )
                location_rows.append(
                    {
                        **base,
                        "location": location,
                        "metric": metric_id,
                        "value": value,
                        "metric_error": error,
                    }
                )

    return global_rows, horizon_rows, location_rows


def metric_direction(metric_id: str) -> str:
    try:
        return str(get_optimization_direction(metric_id).value)
    except Exception:
        # All requested metrics in this experiment are error/loss metrics.
        return "minimize"


# ---------------------------------------------------------------------------
# NetCDF HPO metadata and trial reconstruction
# ---------------------------------------------------------------------------


def read_netcdf_metadata(nc_path: Path) -> dict[str, Any]:
    with xr.open_dataset(nc_path) as ds:
        attrs = dict(ds.attrs)
    result: dict[str, Any] = {
        "model_name_attr": attrs.get("model_name"),
        "model_version_attr": attrs.get("model_version"),
        "chap_version_attr": attrs.get("chap_version"),
        "model_configuration_attr": attrs.get("model_configuration"),
        "model_info_attr": attrs.get("model_info"),
    }
    raw_hpo = attrs.get("hpo")
    if raw_hpo:
        try:
            result["hpo"] = json.loads(raw_hpo)
        except Exception as exc:
            result["hpo_parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def canonical_param_key(row: pd.Series, param_columns: Sequence[str]) -> str:
    payload = {}
    for column in param_columns:
        value = row[column]
        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        payload[column] = value
    return json_dumps(payload)


def reconstruct_trial_history(
    leaderboard_path: Path,
    *,
    run_row: dict[str, Any],
    hpo_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(leaderboard_path)
    if "trial_nr" not in df.columns:
        raise ValueError(f"Missing trial_nr in {leaderboard_path}")

    # HyperparameterOptimization.write_leaderboard writes the already score-ranked
    # leaderboard. trial_nr is therefore the authoritative chronology field.
    df = df.sort_values("trial_nr", kind="stable").reset_index(drop=True)

    fixed_columns = {"trial_nr", "score", "seconds", "failure"}
    param_columns = [c for c in df.columns if c not in fixed_columns]

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    failure_text = df["failure"].fillna("").astype(str).str.strip()
    df["success"] = df["score"].notna() & failure_text.eq("")
    df["failure_type"] = np.where(
        df["success"],
        None,
        failure_text.str.split(":", n=1).str[0].replace("", "unknown"),
    )
    df["cumulative_trial_seconds"] = df["seconds"].fillna(0.0).cumsum()

    direction = str(hpo_meta.get("direction") or metric_direction(run_row["objective_metric"]))
    successful_scores = df["score"].where(df["success"])
    if direction == "maximize":
        df["best_so_far"] = successful_scores.cummax()
    else:
        df["best_so_far"] = successful_scores.cummin()
    df["best_so_far"] = df["best_so_far"].ffill()

    df["param_key"] = df.apply(
        lambda row: canonical_param_key(row, param_columns), axis=1
    )
    df["is_duplicate_config"] = df["param_key"].duplicated(keep="first")

    for key in (
        "run_id",
        "model",
        "dataset",
        "searcher",
        "budget",
        "seed",
        "objective_metric",
    ):
        df[key] = run_row.get(key)
    df["direction"] = direction

    summary = summarize_trial_history(
        df,
        run_row=run_row,
        hpo_meta=hpo_meta,
        param_columns=param_columns,
        direction=direction,
    )
    return df, summary


def summarize_trial_history(
    df: pd.DataFrame,
    *,
    run_row: dict[str, Any],
    hpo_meta: dict[str, Any],
    param_columns: Sequence[str],
    direction: str,
) -> dict[str, Any]:
    successful = df.loc[df["success"]].copy()
    failed = df.loc[~df["success"]].copy()

    base = {
        "run_id": run_row["run_id"],
        "model": run_row["model"],
        "dataset": run_row["dataset"],
        "searcher": run_row.get("searcher"),
        "budget": run_row.get("budget"),
        "seed": run_row.get("seed"),
        "objective_metric": run_row["objective_metric"],
        "direction": direction,
        "n_trials_attempted": int(len(df)),
        "n_successful_trials": int(len(successful)),
        "n_failed_trials": int(len(failed)),
        "failure_rate_pct": float((~df["success"]).mean() * 100) if len(df) else None,
        "n_unique_configs": int(df["param_key"].nunique(dropna=False)) if len(df) else 0,
        "duplicate_config_rate_pct": float(df["is_duplicate_config"].mean() * 100)
        if len(df)
        else None,
        "sum_trial_seconds": float(df["seconds"].fillna(0.0).sum()),
        "median_trial_seconds": float(df["seconds"].dropna().median())
        if df["seconds"].notna().any()
        else None,
        "p95_trial_seconds": float(df["seconds"].dropna().quantile(0.95))
        if df["seconds"].notna().any()
        else None,
        "hpo_seconds": safe_float(hpo_meta.get("seconds")),
        "hpo_best_score": safe_float(hpo_meta.get("best_score")),
        "hpo_best_params": json_dumps(hpo_meta.get("best_params", {})),
        "hpo_model_configuration": json_dumps(hpo_meta.get("model_configuration", {})),
        "search_space": json_dumps(hpo_meta.get("search_space", {})),
        "stop_reason": hpo_meta.get("stop_reason"),
        "param_columns": json_dumps(list(param_columns)),
    }

    if base["hpo_seconds"] not in (None, 0):
        base["trial_time_fraction_of_hpo_pct"] = (
            base["sum_trial_seconds"] / base["hpo_seconds"] * 100
        )
        base["hpo_overhead_seconds"] = max(
            0.0, base["hpo_seconds"] - base["sum_trial_seconds"]
        )
    else:
        base["trial_time_fraction_of_hpo_pct"] = None
        base["hpo_overhead_seconds"] = None

    if successful.empty:
        return base

    best_idx = (
        successful["score"].idxmax()
        if direction == "maximize"
        else successful["score"].idxmin()
    )
    best = df.loc[best_idx]
    base["best_trial_nr"] = int(best["trial_nr"])
    base["time_to_best_seconds"] = float(best["cumulative_trial_seconds"])
    base["first_success_trial_nr"] = int(successful.iloc[0]["trial_nr"])
    base["first_success_score"] = float(successful.iloc[0]["score"])

    first_score = float(successful.iloc[0]["score"])
    best_score = float(best["score"])
    base["observed_best_score"] = best_score

    if direction == "maximize":
        total_gain = best_score - first_score
        target = first_score + 0.95 * total_gain
        reached = df.loc[df["best_so_far"].ge(target)]
    else:
        total_gain = first_score - best_score
        target = first_score - 0.95 * total_gain
        reached = df.loc[df["best_so_far"].le(target)]

    base["objective_improvement_from_first"] = total_gain
    if len(reached):
        r = reached.iloc[0]
        base["trial_to_95pct_improvement"] = int(r["trial_nr"])
        base["seconds_to_95pct_improvement"] = float(r["cumulative_trial_seconds"])
    else:
        base["trial_to_95pct_improvement"] = None
        base["seconds_to_95pct_improvement"] = None

    return base


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------


def add_normal_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    normal = metrics_df.loc[metrics_df["mode"] == "normal", [
        "model", "dataset", "metric", "value"
    ]].rename(columns={"value": "normal_value"})

    hpo = metrics_df.loc[metrics_df["mode"] == "hpo"].copy()
    merged = hpo.merge(normal, on=["model", "dataset", "metric"], how="left")
    merged["delta_hpo_minus_normal"] = merged["value"] - merged["normal_value"]

    directions = merged["metric"].map(metric_direction)
    denominator = merged["normal_value"].abs().replace(0, np.nan)
    merged["relative_change_pct"] = (
        merged["delta_hpo_minus_normal"] / denominator * 100
    )
    merged["improvement_pct"] = np.where(
        directions.eq("maximize"),
        (merged["value"] - merged["normal_value"]) / denominator * 100,
        (merged["normal_value"] - merged["value"]) / denominator * 100,
    )
    merged["better_than_normal"] = merged["improvement_pct"] > 0
    return merged


def add_location_normal_comparisons(location_df: pd.DataFrame) -> pd.DataFrame:
    if location_df.empty:
        return pd.DataFrame()
    normal = location_df.loc[location_df["mode"] == "normal", [
        "model", "dataset", "location", "metric", "value"
    ]].rename(columns={"value": "normal_value"})
    hpo = location_df.loc[location_df["mode"] == "hpo"].copy()
    merged = hpo.merge(
        normal,
        on=["model", "dataset", "location", "metric"],
        how="left",
    )
    denominator = merged["normal_value"].abs().replace(0, np.nan)
    direction = merged["metric"].map(metric_direction)
    merged["improvement_pct"] = np.where(
        direction.eq("maximize"),
        (merged["value"] - merged["normal_value"]) / denominator * 100,
        (merged["normal_value"] - merged["value"]) / denominator * 100,
    )
    return merged


def location_heterogeneity_summary(location_comparison: pd.DataFrame) -> pd.DataFrame:
    if location_comparison.empty:
        return pd.DataFrame()
    group_cols = ["model", "dataset", "searcher", "budget", "seed", "metric"]
    return (
        location_comparison.groupby(group_cols, dropna=False)["improvement_pct"]
        .agg(
            n_locations="count",
            mean_improvement_pct="mean",
            median_improvement_pct="median",
            sd_improvement_pct="std",
            q10_improvement_pct=lambda x: x.quantile(0.10),
            q90_improvement_pct=lambda x: x.quantile(0.90),
            fraction_locations_improved=lambda x: float((x > 0).mean()),
        )
        .reset_index()
    )


def merge_outer_objective(
    hpo_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    if hpo_summary.empty:
        return hpo_summary.copy()

    objective = comparison.loc[
        comparison["metric"] == comparison["objective_metric"],
        [
            "run_id",
            "value",
            "normal_value",
            "improvement_pct",
            "delta_hpo_minus_normal",
        ],
    ].rename(
        columns={
            "value": "outer_objective_score",
            "normal_value": "normal_objective_score",
            "improvement_pct": "outer_objective_improvement_pct",
            "delta_hpo_minus_normal": "outer_objective_delta",
        }
    )

    runtime = manifest[["run_id", "cli_seconds", "status"]].copy()
    merged = hpo_summary.merge(objective, on="run_id", how="left").merge(
        runtime, on="run_id", how="left"
    )

    merged["selection_generalization_gap"] = (
        merged["outer_objective_score"] - merged["hpo_best_score"]
    )
    denom = merged["hpo_best_score"].abs().replace(0, np.nan)
    merged["selection_generalization_gap_pct"] = (
        merged["selection_generalization_gap"] / denom * 100
    )
    merged["outer_eval_plus_setup_seconds"] = (
        merged["cli_seconds"] - merged["hpo_seconds"]
    )
    return merged


def best_at_cutoff(trace: pd.DataFrame, cutoff: int) -> float | None:
    eligible = trace.loc[(trace["trial_nr"] < cutoff) & trace["success"]]
    if eligible.empty:
        return None
    direction = str(trace["direction"].iloc[0])
    if direction == "maximize":
        return float(eligible["score"].max())
    return float(eligible["score"].min())


def prefix_consistency(
    trials_df: pd.DataFrame,
    hpo_runs_df: pd.DataFrame,
    budgets: Sequence[int],
) -> pd.DataFrame:
    if trials_df.empty or hpo_runs_df.empty:
        return pd.DataFrame()

    max_budget = max(budgets)
    rows: list[dict[str, Any]] = []
    grouping = ["model", "dataset", "searcher", "seed"]
    for keys, group in trials_df.groupby(grouping, dropna=False):
        model, dataset, searcher, seed = keys
        trace100 = group.loc[group["budget"] == max_budget].sort_values("trial_nr")
        if trace100.empty:
            continue
        for cutoff in sorted(b for b in budgets if b < max_budget):
            prefix_score = best_at_cutoff(trace100, cutoff)
            actual = hpo_runs_df.loc[
                (hpo_runs_df["model"] == model)
                & (hpo_runs_df["dataset"] == dataset)
                & (hpo_runs_df["searcher"] == searcher)
                & (hpo_runs_df["seed"] == seed)
                & (hpo_runs_df["budget"] == cutoff),
                "hpo_best_score",
            ]
            actual_score = safe_float(actual.iloc[0]) if len(actual) else None
            abs_diff = (
                abs(prefix_score - actual_score)
                if prefix_score is not None and actual_score is not None
                else None
            )
            scale = max(
                1.0,
                abs(prefix_score or 0.0),
                abs(actual_score or 0.0),
            )
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "searcher": searcher,
                    "seed": seed,
                    "cutoff_budget": cutoff,
                    "max_budget": max_budget,
                    "prefix_best_score": prefix_score,
                    "independent_budget_best_score": actual_score,
                    "absolute_difference": abs_diff,
                    "relative_difference": abs_diff / scale if abs_diff is not None else None,
                    "prefix_match_rtol_1e-8": (
                        bool(abs_diff <= 1e-8 * scale)
                        if abs_diff is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def convergence_efficiency(
    trials_df: pd.DataFrame, budgets: Sequence[int]
) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()
    max_budget = max(budgets)
    rows: list[dict[str, Any]] = []
    grouping = ["model", "dataset", "searcher", "seed"]
    for keys, group in trials_df.groupby(grouping, dropna=False):
        trace = group.loc[group["budget"] == max_budget].sort_values("trial_nr")
        successful = trace.loc[trace["success"]]
        if successful.empty:
            continue
        model, dataset, searcher, seed = keys
        direction = str(trace["direction"].iloc[0])
        first = float(successful.iloc[0]["score"])
        full = best_at_cutoff(trace, max_budget)
        if full is None:
            continue
        total_gain = full - first if direction == "maximize" else first - full
        for cutoff in sorted(set(budgets)):
            score = best_at_cutoff(trace, cutoff)
            if score is None:
                fraction = None
            else:
                gain = score - first if direction == "maximize" else first - score
                fraction = gain / total_gain if total_gain > 0 else 1.0
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "searcher": searcher,
                    "seed": seed,
                    "cutoff": cutoff,
                    "best_score_at_cutoff": score,
                    "first_success_score": first,
                    "full_budget_best_score": full,
                    "fraction_of_full_improvement": fraction,
                }
            )
    return pd.DataFrame(rows)


def parameter_failure_summary(
    trials_df: pd.DataFrame,
    budgets: Sequence[int],
) -> pd.DataFrame:
    """Describe where trial failures concentrate, using max-budget traces only.

    Using only the maximum budget avoids counting the same seeded prefix three
    times in the 20/50/100 design.
    """
    if trials_df.empty:
        return pd.DataFrame()
    max_budget = max(budgets)
    data = trials_df.loc[trials_df["budget"] == max_budget].copy()
    fixed = {
        "trial_nr",
        "score",
        "seconds",
        "failure",
        "success",
        "failure_type",
        "cumulative_trial_seconds",
        "best_so_far",
        "param_key",
        "is_duplicate_config",
        "run_id",
        "model",
        "dataset",
        "searcher",
        "budget",
        "seed",
        "objective_metric",
        "direction",
    }
    param_columns = [c for c in data.columns if c not in fixed]
    rows: list[dict[str, Any]] = []

    for group_keys, group in data.groupby(
        ["model", "dataset", "searcher"], dropna=False
    ):
        model, dataset, searcher = group_keys
        for param in param_columns:
            if param not in group.columns or group[param].isna().all():
                continue
            series = group[param]
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_fraction = float(numeric.notna().mean())
            n_unique = int(series.nunique(dropna=True))

            if numeric_fraction >= 0.95 and n_unique > 10:
                try:
                    bins = pd.qcut(numeric, q=min(5, n_unique), duplicates="drop")
                except ValueError:
                    bins = pd.Series(["all"] * len(group), index=group.index)
                labels = bins.astype(str)
                kind = "numeric_quantile_bin"
            else:
                labels = series.astype(str)
                kind = "categorical_or_discrete"

            tmp = group.assign(_level=labels)
            for level, level_df in tmp.groupby("_level", dropna=False):
                scores = level_df.loc[level_df["success"], "score"]
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "searcher": searcher,
                        "parameter": param,
                        "parameter_kind": kind,
                        "level_or_bin": str(level),
                        "n_trials": int(len(level_df)),
                        "n_failures": int((~level_df["success"]).sum()),
                        "failure_rate_pct": float((~level_df["success"]).mean() * 100),
                        "mean_success_score": float(scores.mean()) if len(scores) else None,
                        "median_success_score": float(scores.median()) if len(scores) else None,
                    }
                )
    return pd.DataFrame(rows)


def failure_type_summary(trials_df: pd.DataFrame, budgets: Sequence[int]) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()
    max_budget = max(budgets)
    failed = trials_df.loc[
        (trials_df["budget"] == max_budget) & (~trials_df["success"])
    ].copy()
    if failed.empty:
        return pd.DataFrame()
    return (
        failed.groupby(["model", "dataset", "searcher", "failure_type"], dropna=False)
        .size()
        .rename("n_failures")
        .reset_index()
    )


def parse_best_params(hpo_runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in hpo_runs_df.iterrows():
        try:
            params = json.loads(row["hpo_best_params"])
        except Exception:
            continue
        for name, value in params.items():
            rows.append(
                {
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "searcher": row["searcher"],
                    "budget": row["budget"],
                    "seed": row["seed"],
                    "parameter": name,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def hyperparameter_stability(
    best_params_long: pd.DataFrame, budgets: Sequence[int]
) -> pd.DataFrame:
    if best_params_long.empty:
        return pd.DataFrame()
    max_budget = max(budgets)
    data = best_params_long.loc[best_params_long["budget"] == max_budget].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(
        ["model", "dataset", "searcher", "parameter"], dropna=False
    ):
        model, dataset, searcher, parameter = keys
        numeric = pd.to_numeric(group["value"], errors="coerce")
        if float(numeric.notna().mean()) >= 0.95:
            mean = float(numeric.mean())
            std = float(numeric.std(ddof=1)) if len(numeric.dropna()) > 1 else 0.0
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "searcher": searcher,
                    "parameter": parameter,
                    "kind": "numeric",
                    "n": int(numeric.notna().sum()),
                    "n_unique": int(numeric.nunique(dropna=True)),
                    "mean": mean,
                    "std": std,
                    "cv_abs": abs(std / mean) if mean != 0 else None,
                    "min": float(numeric.min()),
                    "median": float(numeric.median()),
                    "max": float(numeric.max()),
                    "mode": None,
                    "mode_frequency": None,
                }
            )
        else:
            values = group["value"].astype(str)
            counts = values.value_counts(dropna=False)
            mode = str(counts.index[0]) if len(counts) else None
            mode_frequency = float(counts.iloc[0] / counts.sum()) if len(counts) else None
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "searcher": searcher,
                    "parameter": parameter,
                    "kind": "categorical",
                    "n": int(len(values)),
                    "n_unique": int(values.nunique(dropna=False)),
                    "mean": None,
                    "std": None,
                    "cv_abs": None,
                    "min": None,
                    "median": None,
                    "max": None,
                    "mode": mode,
                    "mode_frequency": mode_frequency,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 2026,
) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return None, None, None
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def paired_searcher_comparison(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare TPE and random on final outer evaluation.

    Primary paired outcome is percentage improvement over the normal baseline on
    each model's own objective metric. This avoids pooling raw RMSE and CRPS-log1p
    values across different models/datasets.
    """
    if comparison.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = comparison.loc[
        comparison["metric"] == comparison["objective_metric"],
        [
            "model",
            "dataset",
            "budget",
            "seed",
            "searcher",
            "value",
            "improvement_pct",
        ],
    ].copy()
    pivot = data.pivot_table(
        index=["model", "dataset", "budget", "seed"],
        columns="searcher",
        values=["value", "improvement_pct"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    required = {
        "value_random",
        "value_tpe",
        "improvement_pct_random",
        "improvement_pct_tpe",
    }
    if not required.issubset(pivot.columns):
        return pivot, pd.DataFrame()

    pivot["tpe_minus_random_objective"] = pivot["value_tpe"] - pivot["value_random"]
    pivot["tpe_minus_random_improvement_pct"] = (
        pivot["improvement_pct_tpe"] - pivot["improvement_pct_random"]
    )
    pivot["tpe_better_outer_objective"] = pivot["value_tpe"] < pivot["value_random"]

    agg_rows: list[dict[str, Any]] = []
    for keys, group in pivot.groupby(["model", "budget"], dropna=False):
        model, budget = keys
        diffs = group["tpe_minus_random_improvement_pct"].dropna().to_numpy(float)
        mean, low, high = bootstrap_mean_ci(diffs)
        pvalue = None
        if len(diffs) >= 5 and np.any(np.abs(diffs) > 0):
            try:
                from scipy.stats import wilcoxon

                pvalue = float(wilcoxon(diffs).pvalue)
            except Exception:
                pvalue = None
        agg_rows.append(
            {
                "model": model,
                "budget": budget,
                "n_pairs": int(len(diffs)),
                "mean_tpe_minus_random_improvement_pct": mean,
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "median_tpe_minus_random_improvement_pct": float(np.median(diffs))
                if len(diffs)
                else None,
                "tpe_outer_win_rate": float(group["tpe_better_outer_objective"].mean()),
                "wilcoxon_p_exploratory": pvalue,
            }
        )
    return pivot, pd.DataFrame(agg_rows)




def canonical_json_value(value: Any) -> str | None:
    """Normalize JSON-like strings/objects for metadata-consistency checks."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return value.strip()
    try:
        return json_dumps(parsed)
    except Exception:
        return str(value)


def configuration_metadata_audit(
    hpo_runs: pd.DataFrame, netcdf_meta: pd.DataFrame
) -> pd.DataFrame:
    """Check whether the ordinary NetCDF config attr equals the HPO-selected config.

    In the current CLI path the standard ``model_configuration`` attribute is written
    from the user/base configuration, while the selected HPO configuration is stored
    inside the flattened HPO metadata.  This audit makes that distinction explicit.
    """
    if hpo_runs.empty or netcdf_meta.empty:
        return pd.DataFrame()
    cols = [c for c in ["run_id", "model_configuration_attr"] if c in netcdf_meta.columns]
    if "run_id" not in cols or "model_configuration_attr" not in cols:
        return pd.DataFrame()
    merged = hpo_runs[[
        "run_id", "model", "dataset", "searcher", "budget", "seed",
        "hpo_model_configuration", "hpo_best_params"
    ]].merge(netcdf_meta[cols], on="run_id", how="left")
    merged["standard_config_normalized"] = merged["model_configuration_attr"].map(canonical_json_value)
    merged["selected_config_normalized"] = merged["hpo_model_configuration"].map(canonical_json_value)
    merged["standard_attr_equals_selected_hpo_config"] = (
        merged["standard_config_normalized"] == merged["selected_config_normalized"]
    )
    return merged


def budget_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    objectives = comparison.loc[
        comparison["metric"] == comparison["objective_metric"]
    ].copy()
    return (
        objectives.groupby(["model", "dataset", "searcher", "budget"], dropna=False)
        .agg(
            n_runs=("value", "count"),
            mean_outer_objective=("value", "mean"),
            sd_outer_objective=("value", "std"),
            median_outer_objective=("value", "median"),
            mean_improvement_over_normal_pct=("improvement_pct", "mean"),
            median_improvement_over_normal_pct=("improvement_pct", "median"),
            fraction_runs_better_than_normal=("better_than_normal", "mean"),
        )
        .reset_index()
    )


def runtime_scaling(hpo_runs: pd.DataFrame) -> pd.DataFrame:
    if hpo_runs.empty:
        return pd.DataFrame()
    return (
        hpo_runs.groupby(["model", "dataset", "searcher", "budget"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            mean_hpo_seconds=("hpo_seconds", "mean"),
            sd_hpo_seconds=("hpo_seconds", "std"),
            mean_cli_seconds=("cli_seconds", "mean"),
            mean_seconds_per_attempt=("median_trial_seconds", "mean"),
            mean_failure_rate_pct=("failure_rate_pct", "mean"),
            mean_duplicate_rate_pct=("duplicate_config_rate_pct", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"Plotting disabled: {type(exc).__name__}: {exc}")
        return None


def plot_convergence_by_trial(
    trials_df: pd.DataFrame,
    figures_dir: Path,
    budgets: Sequence[int],
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or trials_df.empty:
        return
    max_budget = max(budgets)
    data = trials_df.loc[trials_df["budget"] == max_budget].copy()
    figures_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), group in data.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for searcher, sgroup in group.groupby("searcher"):
            curves = []
            for _, trace in sgroup.groupby("seed"):
                trace = trace.sort_values("trial_nr")
                series = trace.set_index("trial_nr")["best_so_far"]
                curves.append(series)
            if not curves:
                continue
            frame = pd.concat(curves, axis=1).sort_index().ffill()
            median = frame.median(axis=1, skipna=True)
            q25 = frame.quantile(0.25, axis=1)
            q75 = frame.quantile(0.75, axis=1)
            line = ax.plot(median.index + 1, median.values, label=searcher)[0]
            ax.fill_between(
                median.index + 1,
                q25.values,
                q75.values,
                alpha=0.18,
                color=line.get_color(),
            )
        ax.set_title(f"Best-so-far convergence: {model} / {dataset}")
        ax.set_xlabel("Attempted trial")
        ax.set_ylabel("Best objective score so far")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Searcher")
        fig.tight_layout()
        fig.savefig(figures_dir / f"convergence_trial__{model}__{dataset}.png", dpi=180)
        plt.close(fig)


def plot_convergence_by_time(
    trials_df: pd.DataFrame,
    figures_dir: Path,
    budgets: Sequence[int],
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or trials_df.empty:
        return
    max_budget = max(budgets)
    data = trials_df.loc[trials_df["budget"] == max_budget].copy()
    figures_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), group in data.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for (searcher, seed), trace in group.groupby(["searcher", "seed"]):
            trace = trace.sort_values("trial_nr")
            ax.plot(
                trace["cumulative_trial_seconds"] / 60.0,
                trace["best_so_far"],
                alpha=0.65,
                label=f"{searcher}, seed={seed}",
            )
        ax.set_title(f"Time-based convergence: {model} / {dataset}")
        ax.set_xlabel("Cumulative objective-evaluation time (minutes)")
        ax.set_ylabel("Best objective score so far")
        ax.grid(True, alpha=0.25)
        if group["seed"].nunique() <= 5:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"convergence_time__{model}__{dataset}.png", dpi=180)
        plt.close(fig)


def plot_budget_outer_performance(
    comparison: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or comparison.empty:
        return
    data = comparison.loc[
        comparison["metric"] == comparison["objective_metric"]
    ].copy()
    figures_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), group in data.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for searcher, sgroup in group.groupby("searcher"):
            summary = (
                sgroup.groupby("budget")["improvement_pct"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("budget")
            )
            ax.errorbar(
                summary["budget"],
                summary["mean"],
                yerr=summary["std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=searcher,
            )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"Outer-evaluation gain vs HPO budget: {model} / {dataset}")
        ax.set_xlabel("Maximum attempted HPO trials")
        ax.set_ylabel("Improvement over normal baseline (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Searcher")
        fig.tight_layout()
        fig.savefig(figures_dir / f"budget_gain__{model}__{dataset}.png", dpi=180)
        plt.close(fig)


def plot_runtime_scaling(hpo_runs: pd.DataFrame, figures_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or hpo_runs.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    for (model, dataset), group in hpo_runs.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for searcher, sgroup in group.groupby("searcher"):
            summary = (
                sgroup.groupby("budget")["hpo_seconds"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("budget")
            )
            ax.errorbar(
                summary["budget"],
                summary["mean"] / 60.0,
                yerr=summary["std"].fillna(0.0) / 60.0,
                marker="o",
                capsize=3,
                label=searcher,
            )
        ax.set_title(f"HPO runtime scaling: {model} / {dataset}")
        ax.set_xlabel("Maximum attempted HPO trials")
        ax.set_ylabel("HPO time (minutes)")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Searcher")
        fig.tight_layout()
        fig.savefig(figures_dir / f"runtime_scaling__{model}__{dataset}.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data available._"
    show = df.head(max_rows).copy()
    try:
        return show.to_markdown(index=False, floatfmt=".4g")
    except Exception:
        return "```text\n" + show.to_string(index=False) + "\n```"


def describe_manifest(manifest: pd.DataFrame) -> str:
    if manifest.empty:
        return "No runs were discovered."
    counts = manifest["status"].value_counts(dropna=False).to_dict()
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def generate_report(
    root: Path,
    *,
    manifest: pd.DataFrame,
    dataset_profiles: pd.DataFrame,
    comparison: pd.DataFrame,
    hpo_runs: pd.DataFrame,
    budget_df: pd.DataFrame,
    searcher_agg: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prefix_df: pd.DataFrame,
    failure_types: pd.DataFrame,
    stability_df: pd.DataFrame,
    location_summary: pd.DataFrame,
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> None:
    max_budget = max(budgets)

    objective_summary = comparison.loc[
        comparison["metric"] == comparison["objective_metric"]
    ].copy()
    if not objective_summary.empty:
        objective_summary = (
            objective_summary.groupby(
                ["model", "dataset", "searcher", "budget"], dropna=False
            )
            .agg(
                mean_outer_score=("value", "mean"),
                mean_normal_score=("normal_value", "mean"),
                mean_improvement_pct=("improvement_pct", "mean"),
                sd_improvement_pct=("improvement_pct", "std"),
                fraction_better_than_normal=("better_than_normal", "mean"),
            )
            .reset_index()
        )

    hpo_diag = hpo_runs[[
        c
        for c in [
            "model",
            "dataset",
            "searcher",
            "budget",
            "seed",
            "n_trials_attempted",
            "n_failed_trials",
            "failure_rate_pct",
            "duplicate_config_rate_pct",
            "best_trial_nr",
            "time_to_best_seconds",
            "trial_to_95pct_improvement",
            "hpo_seconds",
            "outer_objective_improvement_pct",
            "selection_generalization_gap_pct",
        ]
        if c in hpo_runs.columns
    ]].copy() if not hpo_runs.empty else pd.DataFrame()

    prefix_rate = None
    if not prefix_df.empty and "prefix_match_rtol_1e-8" in prefix_df:
        valid = prefix_df["prefix_match_rtol_1e-8"].dropna()
        prefix_rate = float(valid.mean()) if len(valid) else None

    lines = [
        "# CHAP hyperparameter optimization analysis",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Experimental design",
        "",
        (
            f"The experiment evaluates two model templates on three monthly datasets using "
            f"a {N_SPLITS}-split rolling-origin backtest, prediction horizon {N_PERIODS}, and "
            f"stride {STRIDE}. HPO is evaluated with RandomSearcher and TPESearcher at budgets "
            f"{list(budgets)} and seeds {list(seeds)}. The model-specific objective is RMSE for "
            f"minimal_template_example and CRPS-log1p for auto_regressive_monthly_v2; every final "
            f"outer evaluation is additionally assessed with {list(REQUESTED_METRICS)}."
        ),
        "",
        (
            "The CLI path is the system under test. HPO tunes on the initial training portion "
            "created by the outer Evaluation.create call; the selected configuration is then "
            "evaluated on the outer rolling-origin forecast windows. This separates model selection "
            "from the final evaluation windows more appropriately than tuning directly on the same "
            "forecast targets used for reporting."
        ),
        "",
        "Run status: " + describe_manifest(manifest),
        "",
        "## Dataset profiles",
        "",
        markdown_table(dataset_profiles),
        "",
        "## Primary result: normal vs HPO on each model's objective",
        "",
        markdown_table(objective_summary, max_rows=80),
        "",
        "Positive `mean_improvement_pct` means lower loss than the corresponding normal-mode baseline.",
        "",
        "## Budget experiment",
        "",
        markdown_table(budget_df, max_rows=80),
        "",
        (
            "Budget effects should be interpreted using final outer-evaluation metrics, not only the "
            "internal best trial score. More trials can improve the inner objective while leaving "
            "outer performance unchanged or worse because of selection noise / overfitting."
        ),
        "",
        "## TPE vs random",
        "",
        markdown_table(searcher_agg, max_rows=40),
        "",
        (
            "The paired comparison uses the difference in percentage improvement over each normal "
            "baseline on the model-specific objective. Bootstrap intervals and Wilcoxon p-values are "
            "exploratory: repeated seeds and datasets are not guaranteed to be independent samples, "
            "and small numbers of datasets limit inferential strength."
        ),
        "",
        "## Convergence efficiency from the maximum-budget traces",
        "",
        markdown_table(convergence_df, max_rows=80),
        "",
        "## HPO diagnostics",
        "",
        markdown_table(hpo_diag, max_rows=80),
        "",
        "## Cross-budget prefix reproducibility",
        "",
        markdown_table(prefix_df, max_rows=60),
        "",
        (
            f"Observed prefix-match rate: {prefix_rate:.3f}."
            if prefix_rate is not None
            else "No prefix-consistency statistic was available."
        ),
        "",
        (
            "With the same seed, random search should normally reproduce the same early candidate "
            "sequence, and TPE should do the same if objective values are deterministic and software "
            "state is unchanged. A mismatch between an independent 20/50-trial run and the matching "
            f"prefix of the {max_budget}-trial run is therefore a useful reproducibility warning."
        ),
        "",
        "## Trial failures",
        "",
        markdown_table(failure_types, max_rows=60),
        "",
        (
            "The optimizer counts attempted trials toward max_trials even when a trial fails. Therefore "
            "a nominal budget of 100 is an attempt budget, not necessarily 100 successful objective "
            "evaluations. Failure-rate tables and search-space hot-spot tables should be inspected "
            "before attributing performance differences solely to the search algorithm."
        ),
        "",
        "## Best-parameter stability at the maximum budget",
        "",
        markdown_table(stability_df, max_rows=80),
        "",
        "## Location-level robustness",
        "",
        markdown_table(location_summary, max_rows=80),
        "",
        "## Implementation-level interpretation",
        "",
        textwrap.dedent(
            """
            1. **Trial chronology must be reconstructed explicitly.** The optimizer sorts successful
               trials by objective before returning its leaderboard. The emitted CSV therefore is not
               chronological, but `trial_nr` preserves the original order. All convergence analyses in
               this script sort by `trial_nr` first.
            2. **Random search samples with replacement.** Duplicate configurations are legitimate under
               the current implementation and consume budget. Duplicate-rate reporting quantifies that
               efficiency cost, especially for small/discrete search spaces.
            3. **Failures are first-class observations.** A failed objective is passed to the searcher as
               `None`; TPE records an Optuna FAIL state, while random search simply continues. The analysis
               reports failure types and parameter regions instead of silently dropping them.
            4. **The persisted HPO record is richer than `Evaluation.from_file()`.** NetCDF contains a
               flattened HPO object, but file re-loading does not currently reconstruct a live
               HyperparameterOptimization. This script reads the raw `hpo` NetCDF attribute directly and
               uses `Evaluation.from_file()` only for forecast/observation metric computation.
            5. **Only the objective is recorded per trial.** The current `Trial` schema stores one scalar
               score, time, parameters, and failure text. Consequently, multi-metric analysis is rigorous
               for the final tuned outer evaluations, but not for every candidate configuration. A package
               extension would be needed to persist per-trial CRPS, CRPS-log1p, MAE, and RMSE together.
            6. **Sampler implementation details affect reproducibility.** TPESearcher persists the class
               name and seed but not the full Optuna sampler configuration. Record the Optuna version and,
               for publication-grade reproducibility, consider persisting sampler kwargs (including startup
               behavior) in the HPO metadata.
            7. **The HPO seed seeds the searcher, not the forecasting model.** In the current optimizer,
               `seed` is passed to `searcher.reset(...)`; it is not automatically propagated into the model
               configuration or external estimator. Exact seeded-prefix reproducibility therefore additionally
               requires a deterministic objective/model (or a separately controlled model seed).
            8. **Selection score and outer score answer different questions.** `best_score` is obtained on
               the inner HPO backtest of the outer training data; the final NetCDF metric is from the outer
               evaluation. The reported selection/generalization gap is therefore diagnostic, not an error
               in accounting.
            """
        ).strip(),
        "",
        "## Recommended thesis interpretation",
        "",
        textwrap.dedent(
            """
            Treat the experiment as a repeated, blocked comparison. Model × dataset defines the main task
            block; seed captures stochastic search variation; searcher and budget are experimental factors.
            Emphasize effect sizes and uncertainty rather than a single best run. In particular, report:
            (a) outer objective improvement over the normal baseline, (b) all four requested final metrics,
            (c) convergence speed and time-to-best, (d) failure and duplicate rates, (e) parameter stability,
            and (f) whether additional budget produces consistent outer-evaluation gains. Avoid claiming that
            a lower internal HPO score alone demonstrates better generalization.
            """
        ).strip(),
        "",
        "## Output files",
        "",
        "All machine-readable tables are written under `tables/`; figures are under `figures/`; raw CLI outputs are under `raw/`.",
        "",
    ]
    (root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def analyze_existing_runs(
    root: Path,
    manifest: pd.DataFrame,
    *,
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> None:
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    global_metrics: list[dict[str, Any]] = []
    horizon_metrics: list[dict[str, Any]] = []
    location_metrics: list[dict[str, Any]] = []
    hpo_summaries: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    netcdf_meta_rows: list[dict[str, Any]] = []

    good = manifest.loc[
        manifest["status"].isin(["ok", "cached"]) & manifest["output_file"].notna()
    ]

    for _, run_row_series in good.iterrows():
        run_row = run_row_series.to_dict()
        nc_path = Path(run_row["output_file"])
        if not nc_path.exists():
            continue

        print(f"Analyzing {run_row['run_id']}")
        try:
            metadata = read_netcdf_metadata(nc_path)
            netcdf_meta_rows.append(
                {
                    "run_id": run_row["run_id"],
                    "model": run_row["model"],
                    "dataset": run_row["dataset"],
                    "mode": run_row["mode"],
                    "searcher": run_row.get("searcher"),
                    "budget": run_row.get("budget"),
                    "seed": run_row.get("seed"),
                    **{
                        k: json_dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in metadata.items()
                        if k != "hpo"
                    },
                }
            )

            g, h, l = compute_evaluation_metrics(nc_path, run_row=run_row)
            global_metrics.extend(g)
            horizon_metrics.extend(h)
            location_metrics.extend(l)

            if run_row["mode"] == "hpo":
                hpo_meta = metadata.get("hpo")
                leaderboard = Path(str(run_row["leaderboard_file"]))
                if not isinstance(hpo_meta, dict):
                    print(f"  warning: no parseable HPO metadata in {nc_path}")
                    continue
                if not leaderboard.exists():
                    print(f"  warning: missing leaderboard {leaderboard}")
                    continue
                trial_df, summary = reconstruct_trial_history(
                    leaderboard,
                    run_row=run_row,
                    hpo_meta=hpo_meta,
                )
                trial_frames.append(trial_df)
                hpo_summaries.append(summary)
        except Exception as exc:
            print(
                f"  analysis failure for {run_row['run_id']}: "
                f"{type(exc).__name__}: {exc}"
            )

    metrics_df = pd.DataFrame(global_metrics)
    horizon_df = pd.DataFrame(horizon_metrics)
    location_df = pd.DataFrame(location_metrics)
    trials_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    hpo_summary_df = pd.DataFrame(hpo_summaries)
    netcdf_meta_df = pd.DataFrame(netcdf_meta_rows)

    save_df(metrics_df, tables / "evaluation_metrics_global.csv")
    save_df(horizon_df, tables / "evaluation_metrics_by_horizon.csv")
    save_df(location_df, tables / "evaluation_metrics_by_location.csv")
    save_df(trials_df, tables / "hpo_trial_history.csv")
    save_df(netcdf_meta_df, tables / "netcdf_metadata.csv")

    comparison = add_normal_comparisons(metrics_df)
    save_df(comparison, tables / "normal_vs_hpo_metrics.csv")

    location_comparison = add_location_normal_comparisons(location_df)
    location_summary = location_heterogeneity_summary(location_comparison)
    save_df(location_comparison, tables / "normal_vs_hpo_by_location.csv")
    save_df(location_summary, tables / "location_heterogeneity_summary.csv")

    hpo_runs = merge_outer_objective(hpo_summary_df, comparison, manifest)
    save_df(hpo_runs, tables / "hpo_run_summary.csv")

    config_audit = configuration_metadata_audit(hpo_runs, netcdf_meta_df)
    save_df(config_audit, tables / "configuration_metadata_audit.csv")

    budget_df = budget_summary(comparison)
    save_df(budget_df, tables / "budget_summary.csv")

    runtime_df = runtime_scaling(hpo_runs)
    save_df(runtime_df, tables / "runtime_scaling.csv")

    paired_searcher, searcher_agg = paired_searcher_comparison(comparison)
    save_df(paired_searcher, tables / "tpe_vs_random_paired.csv")
    save_df(searcher_agg, tables / "tpe_vs_random_summary.csv")

    prefix_df = prefix_consistency(trials_df, hpo_runs, budgets)
    save_df(prefix_df, tables / "cross_budget_prefix_consistency.csv")

    convergence_df = convergence_efficiency(trials_df, budgets)
    save_df(convergence_df, tables / "convergence_efficiency.csv")

    failure_types = failure_type_summary(trials_df, budgets)
    failure_params = parameter_failure_summary(trials_df, budgets)
    save_df(failure_types, tables / "failure_types.csv")
    save_df(failure_params, tables / "failure_parameter_hotspots.csv")

    best_params_long = parse_best_params(hpo_runs)
    stability_df = hyperparameter_stability(best_params_long, budgets)
    save_df(best_params_long, tables / "best_hyperparameters_long.csv")
    save_df(stability_df, tables / "hyperparameter_stability.csv")

    plot_convergence_by_trial(trials_df, figures, budgets)
    plot_convergence_by_time(trials_df, figures, budgets)
    plot_budget_outer_performance(comparison, figures)
    plot_runtime_scaling(hpo_runs, figures)

    dataset_profiles_path = tables / "dataset_profiles.csv"
    dataset_profiles = (
        pd.read_csv(dataset_profiles_path)
        if dataset_profiles_path.exists()
        else pd.DataFrame()
    )

    generate_report(
        root,
        manifest=manifest,
        dataset_profiles=dataset_profiles,
        comparison=comparison,
        hpo_runs=hpo_runs,
        budget_df=budget_df,
        searcher_agg=searcher_agg,
        convergence_df=convergence_df,
        prefix_df=prefix_df,
        failure_types=failure_types,
        stability_df=stability_df,
        location_summary=location_summary,
        budgets=budgets,
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
# Validation and argument parsing
# ---------------------------------------------------------------------------


def validate_design(
    models: Sequence[ModelSpec],
    budgets: Sequence[int],
    seeds: Sequence[int],
    *,
    check_files: bool = True,
) -> None:
    if not budgets or any(int(b) <= 0 for b in budgets):
        raise ValueError("All budgets must be positive integers")
    if not seeds:
        raise ValueError("At least one seed is required")
    if not check_files:
        return
    for model in models:
        if model.configuration_yaml is not None and not model.configuration_yaml.exists():
            raise FileNotFoundError(
                f"Base model configuration for {model.key} not found: {model.configuration_yaml}. "
                "Pass --auto-reg-config to a valid base configuration, or omit that option."
            )
        if model.search_space_yaml is None:
            raise ValueError(
                f"Explicit HPO search-space YAML is required for {model.key}. "
                "Pass --auto-reg-search-space or --minimal-search-space as appropriate. "
                "The experiment intentionally does not fall back to the model's MLProject hpo_search_space."
            )
        if not model.search_space_yaml.exists():
            raise FileNotFoundError(
                f"Search-space YAML for {model.key} not found: {model.search_space_yaml}"
            )
        if not model.search_space_yaml.is_file():
            raise ValueError(
                f"Search-space path for {model.key} is not a file: {model.search_space_yaml}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze CHAP normal/HPO evaluations across models, datasets, searchers, and budgets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "results" / "hpo_master_analysis",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(BASE_MODEL_SPECS),
        default=list(BASE_MODEL_SPECS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument(
        "--searchers",
        nargs="+",
        choices=["random", "tpe"],
        default=list(DEFAULT_SEARCHERS),
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Repeated seeds. Use one seed for a cheaper smoke run; >=5 is preferable for stronger stochastic-search inference.",
    )
    parser.add_argument(
        "--chap-command",
        default=None,
        help="Override CHAP executable, e.g. \"uv run chap\". Default prefers the sibling chap-core checkout.",
    )
    parser.add_argument(
        "--auto-reg-config",
        type=Path,
        default=None,
        help=(
            "Optional base model configuration for auto_regressive_monthly_v2. "
            "This is separate from the HPO search space and, when supplied, is passed via "
            "--model-configuration-yaml."
        ),
    )
    parser.add_argument(
        "--auto-reg-search-space",
        type=Path,
        default=AUTO_REG_DEFAULT_SEARCH_SPACE,
        help=(
            "HPO search-space YAML for auto_regressive_monthly_v2. Default: "
            "scripts/auto_reg_monthly_v2_conf.yaml. Passed through "
            "--estimator-options.search-space."
        ),
    )
    parser.add_argument(
        "--minimal-search-space",
        type=Path,
        default=MINIMAL_DEFAULT_SEARCH_SPACE,
        help=(
            "HPO search-space YAML for minimal_template_example. Default: "
            "scripts/minimal_template_config_hpo.yaml. Passed through "
            "--estimator-options.search-space."
        ),
    )
    parser.add_argument(
        "--ignore-environment",
        action="store_true",
        help="Append --run-config.ignore-environment to chap eval.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run completed CLI experiments instead of reusing their outputs.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Do not run chap eval; analyze existing outputs in --output-dir.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed chap eval instead of recording the failure and continuing.",
    )
    parser.add_argument(
        "--skip-dataset-profile",
        action="store_true",
        help="Skip pandas-based dataset profiling (useful if offline and raw URLs are unavailable).",
    )
    return parser.parse_args()


def discover_manifest_from_design(root: Path, specs: Sequence[RunSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        paths = output_paths(root, spec)
        metadata = load_json(paths["metadata"]) or {}
        exists = completed_run_exists(paths, spec.mode)
        rows.append(
            {
                "run_id": spec.run_id,
                "model": spec.model.key,
                "dataset": spec.dataset.key,
                "mode": spec.mode,
                "searcher": spec.searcher,
                "budget": spec.budget,
                "seed": spec.seed,
                "objective_metric": spec.model.objective_metric,
                "search_space_yaml": str(spec.model.search_space_yaml) if spec.mode == "hpo" else None,
                "status": "cached" if exists else "missing",
                "returncode": metadata.get("returncode"),
                "cli_seconds": metadata.get("seconds"),
                "output_file": str(paths["nc"]),
                "leaderboard_file": str(paths["leaderboard"]) if spec.mode == "hpo" else None,
                "log_file": str(paths["log"]),
                "command": paths["command"].read_text(encoding="utf-8").strip()
                if paths["command"].exists()
                else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)

    models = [BASE_MODEL_SPECS[key] for key in args.models]
    configured_models: list[ModelSpec] = []
    for model in models:
        if model.key == "auto_regressive_monthly_v2":
            configured_models.append(
                replace(
                    model,
                    configuration_yaml=args.auto_reg_config.resolve()
                    if args.auto_reg_config
                    else None,
                    search_space_yaml=args.auto_reg_search_space.resolve(),
                )
            )
        elif model.key == "minimal_template_example":
            configured_models.append(
                replace(
                    model,
                    search_space_yaml=args.minimal_search_space.resolve(),
                )
            )
        else:
            configured_models.append(model)
    models = configured_models

    datasets = [DATASET_SPECS[key] for key in args.datasets]
    budgets = sorted(set(int(x) for x in args.budgets))
    seeds = list(dict.fromkeys(int(x) for x in args.seeds))
    searchers = list(dict.fromkeys(args.searchers))

    validate_design(models, budgets, seeds, check_files=not args.analysis_only)
    chap_command = (
        resolve_chap_command(args.chap_command)
        if not args.analysis_only
        else (shlex.split(args.chap_command) if args.chap_command else ["<analysis-only>"])
    )

    specs = build_run_specs(models, datasets, searchers, budgets, seeds)
    n_normal = sum(s.mode == "normal" for s in specs)
    n_hpo = sum(s.mode == "hpo" for s in specs)
    total_trial_attempts = sum(
        int(s.budget or 0) for s in specs if s.mode == "hpo"
    )

    print("CHAP HPO experiment design")
    print(f"  local chap-core: {CHAP_CORE_ROOT}")
    print(f"  chap command:    {shlex.join(chap_command)}")
    print(f"  output:          {root}")
    print(f"  normal runs:     {n_normal}")
    print(f"  HPO runs:        {n_hpo}")
    print(f"  HPO attempts:    {total_trial_attempts:,}")
    print(f"  seeds:           {seeds}")

    write_provenance(
        root,
        chap_command=chap_command,
        models=models,
        datasets=datasets,
        searchers=searchers,
        budgets=budgets,
        seeds=seeds,
    )

    if not args.skip_dataset_profile and not args.analysis_only:
        profiles = pd.DataFrame([profile_dataset(d) for d in datasets])
        save_df(profiles, root / "tables" / "dataset_profiles.csv")

    if args.analysis_only:
        manifest = discover_manifest_from_design(root, specs)
    else:
        manifest_rows: list[dict[str, Any]] = []
        for spec in specs:
            manifest_rows.append(
                execute_run(
                    root=root,
                    spec=spec,
                    chap_command=chap_command,
                    force=args.force,
                    ignore_environment=args.ignore_environment,
                    fail_fast=args.fail_fast,
                )
            )
            # Persist after every run, so an interrupted long experiment resumes cleanly.
            save_df(pd.DataFrame(manifest_rows), root / "tables" / "run_manifest_partial.csv")
        manifest = pd.DataFrame(manifest_rows)

    save_df(manifest, root / "tables" / "run_manifest.csv")
    analyze_existing_runs(root, manifest, budgets=budgets, seeds=seeds)

    print("\nAnalysis complete")
    print(f"  report:  {root / 'analysis_report.md'}")
    print(f"  tables:  {root / 'tables'}")
    print(f"  figures: {root / 'figures'}")
    print(f"  raw:     {root / 'raw'}")


if __name__ == "__main__":
    main()
