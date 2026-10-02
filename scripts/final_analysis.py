#!/usr/bin/env python3
"""Compare normal evaluation with random/TPE HPO for two models and datasets.

Place this file at chap-core/chap_core/hpo/scripts/final_analysis.py and run:
    python chap_core/hpo/scripts/final_analysis.py
    python chap_core/hpo/scripts/final_analysis.py --plan

All paths are resolved relative to chap_core/hpo, regardless of the current
working directory. Results go to chap_core/hpo/results/final_analysis/.
"""

import argparse
import csv
import traceback
from pathlib import Path
from time import perf_counter
import os
import sys

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ANALYSIS_DIR.parent / "chap-core"
RESULTS_DIR = ANALYSIS_DIR / "results" / "final_analysis"
METRICS = ("crps_log1p", "crps", "rmse", "mae", "coverage_10_90", "coverage_25_75", "winkler_score_10_90", "ratio_above_truth", "crps_norm")
MAX_TRIALS = (20, 50, 100)
SEEDS = (17, 42, 123)
MODEL_NAMES = (
    "auto_regressive_monthly_v2",
    "mstl_multistep_model",
    "minimal_template_example",
)


def experiments():
    datasets = {
        "lao": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv", # smallest
        "vnm": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/vnm/chap_VNM_admin1_monthly.csv", # second
        "tha": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/tha/chap_THA_admin1_monthly.csv", # largest
    }
    models = {
        "auto_regressive_monthly_v2": {
            "name": "https://github.com/chap-models/auto_regressive_monthly_v2/",
            "search_space": ANALYSIS_DIR / "search_spaces" / "auto_reg_ss.yaml",
            "objective": "crps_log1p",
            "model_configurations": {
                17: ANALYSIS_DIR / "model_configurations" / "auto_reg_17_mc.yaml",
                42: ANALYSIS_DIR / "model_configurations" / "auto_reg_42_mc.yaml",
                123: ANALYSIS_DIR / "model_configurations" / "auto_reg_123_mc.yaml",
            },
        },
        "mstl_multistep_model": {
            "name": "https://github.com/knutdrand/mstl_multistep_model/",
            "search_space": ANALYSIS_DIR / "search_spaces" / "mstl_ss.yaml",
            "objective": "crps_log1p",
            "model_configurations": {
                17: ANALYSIS_DIR / "model_configurations" / "mstl_17_mc.yaml",
                42: ANALYSIS_DIR / "model_configurations" / "mstl_42_mc.yaml",
                123: ANALYSIS_DIR / "model_configurations" / "mstl_123_mc.yaml",
            },
        },
        "minimal_template_example": {
            # "name": str(REPO_ROOT.parent / "minimal_template_example"),
            "name": "https://github.com/chap-models/minimal_template_example/",
            "search_space": ANALYSIS_DIR / "search_spaces" / "mini_temp_ss.yaml",
            "objective": "rmse",
            "model_configurations": {},
        },
    }

    for seed in SEEDS:
        for model_id, model in models.items():
            for dataset_id, dataset in datasets.items():
                for mode, searcher in (("normal", None), ("hpo", "random"), ("hpo", "tpe")):
                    for max_trials in (None,) if searcher is None else MAX_TRIALS:
                        yield model_id, model, dataset_id, dataset, mode, searcher, max_trials, seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--n-periods", type=int, default=3, help="Forecast horizon (chap default: 3)")
    # parser.add_argument("--n-splits", type=int, default=7, help="Backtest splits (chap default: 7)")
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default=None,
        help="Run only the selected model. If omitted, run all models.",
    )
    parser.add_argument("--plan", action="store_true", help="Print the runs without executing")
    args = parser.parse_args()
    # if args.n_periods < 1 or args.n_splits < 1:
    #     parser.error("--n-periods and --n-splits must be positive")

    # runs = list(experiments())
    runs = [
        run
        for run in experiments()
        if args.model_name is None or run[0] == args.model_name
    ]
    if args.plan:
        for i, (model_id, model, dataset_id, dataset, mode, searcher, max_trials, seed) in enumerate(runs, start=1):
            print(f"{i}. {model_id} | {dataset_id} | {mode} | {searcher or '-'} | "
                  f"{model['objective'] if searcher else '-'}  | max_trials={max_trials or '-'} | "
                  f"seed={seed} | {dataset}")
        return 0

    required_files = {
        model["search_space"] for _, model, _, _, _, _, _, _ in runs
    }
    required_files |= {
        model_configuration
        for _, model, _, _, _, _, _, seed in runs
        if (model_configuration := model["model_configurations"].get(seed)) is not None
    }
    missing = sorted((p for p in required_files if not p.exists()), key=str)
    if missing:
        parser.error("Missing inputs:\n  " + "\n  ".join(map(str, missing)))

    # Import only for actual runs so --plan also works outside a CHAP environment.
    from chap_core.api_types import BacktestParams, EstimatorMode, EstimatorOptions, SearcherType
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.metrics import calculate_metrics
    from chap_core.cli_endpoints.evaluate import eval_cmd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_name = (
        f"log_{args.model_name}.txt"
        if args.model_name is not None
        else "log_all.txt"
    )
    log_file = RESULTS_DIR / log_name
    # log_handle = log_file.open("w", encoding="utf-8", buffering=1)
    log_handle = log_file.open("a", encoding="utf-8", buffering=1)

    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_handle.fileno(), sys.stdout.fileno())
    os.dup2(log_handle.fileno(), sys.stderr.fileno())
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print(f"Log file: {log_file}", flush=True)

    # summary_file = RESULTS_DIR / "summary.csv"
    summary_name = (
        f"summary_{args.model_name}.csv"
        if args.model_name is not None
        else "summary_all.csv"
    )
    summary_file = RESULTS_DIR / summary_name
    columns = ("model", "dataset", "mode", "searcher", "objective", *METRICS, "max_trials", "seed", "eval_runtime_s", "hpo_runtime_s", "output_file", "error")
    failures = 0
    # backtest = BacktestParams(n_splits=args.n_splits, n_periods=args.n_periods)
    backtest_params = BacktestParams(n_periods=3, n_splits=12, stride=3) # prevent validation overfitting, default uses (3, 7, 1)

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for run_number, (model_id, model, dataset_id, dataset, mode, searcher, max_trials, seed) in enumerate(runs, 1):
            model_configuration_yaml = model["model_configurations"].get(seed)
            label = f"{model_id}__{dataset_id}__{searcher or 'normal'}{f'__max_trials_{max_trials}' if max_trials else ''}__seed_{seed}"
            output_file = RESULTS_DIR / f"{label}.nc"
            row = dict(
                model=model_id, 
                dataset=dataset_id, 
                mode=mode,
                searcher=searcher or "", 
                objective=model["objective"] if searcher else "",
                max_trials=max_trials if max_trials is not None else "", 
                seed=seed, 
                eval_runtime_s="",
                hpo_runtime_s="",
                output_file=str(output_file), 
                error=""
            )
            print(f"[{run_number}/{len(runs)}] {label}", flush=True)

            if searcher is None:
                options = EstimatorOptions(
                    mode=EstimatorMode.NORMAL,
                )
            else:
                options = EstimatorOptions(
                    mode=EstimatorMode.HPO,
                    search_space=model["search_space"],
                    metric=model["objective"],
                    searcher=SearcherType(searcher),  # random -> RandomSearcher; tpe -> TPESearcher
                    max_trials=max_trials,
                    seed=seed,
                )

            start = perf_counter()
            try:
                # The same CHAP evaluation entry point for both modes.
                eval_cmd(
                    model_name=model["name"],
                    dataset_csv=dataset,
                    output_file=output_file,
                    model_configuration_yaml=model_configuration_yaml,
                    # backtest_params=backtest,
                    backtest_params=backtest_params,
                    estimator_options=options,
                )
                # scores = calculate_metrics(evaluation=Evaluation.from_file(output_file), metric_ids=list(METRICS))
                evaluation = Evaluation.from_file(output_file)
                scores = calculate_metrics(evaluation=evaluation, metric_ids=list(METRICS))
                if searcher is not None:
                    hpo = evaluation.to_flat().hpo
                    if hpo is None:
                        raise ValueError("HPO metadata missing from evaluation")
                    row["hpo_runtime_s"] = round(hpo.seconds, 3)
                
                row.update({metric: scores.get(metric) for metric in METRICS})
                print("  " + ", ".join(f"{k}={scores.get(k)}" for k in METRICS), flush=True)
            except Exception as exc:
                failures += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
                (RESULTS_DIR / f"{label}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"  FAILED: {row['error']}", flush=True)
            finally:
                runtime = round(perf_counter() - start, 3)
                row["eval_runtime_s"] = runtime
                print(f"  Runtime: {runtime:.3f}s", flush=True)

            writer.writerow(row) # writes no matter evaluation succeeds or failes
            f.flush() # write immediately rather than temporarily in memory, keep partial results even if later runs fail.

    print(f"\nSummary: {summary_file}")
    print("HPO leaderboard CSVs are written next to each successful HPO .nc file.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())