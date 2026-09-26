#!/usr/bin/env python3
"""Compare normal evaluation with random/TPE HPO for two models and datasets.

Place this file at chap-core/chap_core/hpo/scripts/simple_hpo_eval.py and run:
    python chap_core/hpo/scripts/simple_hpo_eval.py
    python chap_core/hpo/scripts/simple_hpo_eval.py --plan

All paths are resolved relative to chap_core/hpo, regardless of the current
working directory. Results go to chap_core/hpo/results/simple_eval/.
"""

import argparse
import csv
import traceback
from pathlib import Path

HPO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = HPO_DIR.parent / "chap-core"
RESULTS_DIR = HPO_DIR / "results" / "simple_eval"
METRICS = ("rmse", "crps_log1p", "mae", "winkler_score_25_75")
MAX_TRIALS = 20
SEED = 17


def experiments():
    # Honor the requested filename when present; current chap-core uses "hydro".
    hydro = REPO_ROOT / "example_data" / "hyrdro_met_subset.csv"
    if not hydro.is_file():
        hydro = REPO_ROOT / "example_data" / "hydro_met_subset.csv"

    datasets = {
        "vietnam_monthly": REPO_ROOT / "example_data" / "vietnam_monthly.csv",
        "hydro_met_subset": hydro,
    }
    models = {
        "minimal_template_example": {
            "name": str(REPO_ROOT.parent / "minimal_template_example"),
            "search_space": HPO_DIR / "search_spaces" / "minimal_template_ss.yaml",
            "objective": "rmse",
        },
        "mstl_multistep_model": {
            "name": "https://github.com/knutdrand/mstl_multistep_model/",
            "search_space": HPO_DIR / "search_spaces" / "mstl_ss.yaml",
            "objective": "crps_log1p",
        },
    }

    for model_id, model in models.items():
        for dataset_id, dataset in datasets.items():
            for mode, searcher in (("normal", None), ("hpo", "random"), ("hpo", "tpe")):
                yield model_id, model, dataset_id, dataset, mode, searcher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=2, help="Backtest splits (default: 2)")
    parser.add_argument("--n-periods", type=int, default=3, help="Forecast horizon (default: 3)")
    parser.add_argument("--plan", action="store_true", help="Print the 12 runs without executing")
    args = parser.parse_args()
    if args.n_splits < 1 or args.n_periods < 1:
        parser.error("--n-splits and --n-periods must be positive")

    runs = list(experiments())
    if args.plan:
        for model_id, model, dataset_id, dataset, mode, searcher in runs:
            print(f"{model_id} | {dataset_id} | {mode} | {searcher or '-'} | "
                  f"objective={model['objective'] if searcher else '-'} | {dataset}")
        return 0

    required_files = {
        dataset for _, _, _, dataset, _, _ in runs
    } | {
        model["search_space"] for _, model, _, _, _, _ in runs
    } | {REPO_ROOT.parent / "minimal_template_example"}
    missing = sorted((p for p in required_files if not p.exists()), key=str)
    if missing:
        parser.error("Missing inputs:\n  " + "\n  ".join(map(str, missing)))

    # Import only for actual runs so --plan also works outside a CHAP environment.
    from chap_core.api_types import BacktestParams, EstimatorMode, EstimatorOptions, SearcherType
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.metrics import calculate_metrics
    from chap_core.cli_endpoints.evaluate import _run_eval

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "summary.csv"
    columns = ("model", "dataset", "mode", "searcher", "objective", *METRICS, "output_file", "error")
    failures = 0
    backtest = BacktestParams(n_splits=args.n_splits, n_periods=args.n_periods)

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for run_number, (model_id, model, dataset_id, dataset, mode, searcher) in enumerate(runs, 1):
            label = f"{model_id}__{dataset_id}__{searcher or 'normal'}"
            output_file = RESULTS_DIR / f"{label}.nc"
            row = dict(model=model_id, dataset=dataset_id, mode=mode,
                       searcher=searcher or "", objective=model["objective"] if searcher else "",
                       output_file=str(output_file), error="")
            print(f"[{run_number}/{len(runs)}] {label}", flush=True)

            if searcher is None:
                options = EstimatorOptions(mode=EstimatorMode.NORMAL)
            else:
                options = EstimatorOptions(
                    mode=EstimatorMode.HPO,
                    search_space=model["search_space"],
                    metric=model["objective"],
                    searcher=SearcherType(searcher),  # random -> RandomSearcher; tpe -> TPESearcher
                    max_trials=MAX_TRIALS,
                    seed=SEED,
                )

            try:
                # The same CHAP evaluation entry point for both modes.
                _run_eval(
                    model_name=model["name"],
                    dataset_csv=dataset,
                    output_file=output_file,
                    backtest_params=backtest,
                    estimator_options=options,
                )
                scores = calculate_metrics(Evaluation.from_file(output_file), list(METRICS))
                row.update({metric: scores.get(metric) for metric in METRICS})
                print("  " + ", ".join(f"{k}={scores.get(k)}" for k in METRICS), flush=True)
            except Exception as exc:
                failures += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
                (RESULTS_DIR / f"{label}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"  FAILED: {row['error']}", flush=True)

            writer.writerow(row)
            f.flush()  # Keep partial results even if a later run fails.

    print(f"\nSummary: {summary_file}")
    print("HPO leaderboard CSVs are written next to each successful HPO .nc file.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())