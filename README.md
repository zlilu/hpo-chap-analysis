# CHAP HPO Analysis

This repository compares standard CHAP model evaluations with hyperparameter optimization (HPO) using random search and Tree-structured Parzen Estimator (TPE) search.

The experiments use two forecasting models (`auto_regressive_monthly_v2` and `mstl_multistep_model`) and monthly datasets for Laos, Vietnam, and Thailand from [dhis2/climate-health-data](https://github.com/dhis2/climate-health-data).

## Experiments

For each model and dataset, the script compares a standard evaluation with HPO using:

- **Search methods:** random search and TPE
- **Trial budgets:** 20, 50, and 100
- **Random seeds:** 42 and 123
- **Optimization metric:** `crps_log1p`
- **Backtesting:** 12 splits, 3 forecast periods, stride 1

This produces 84 evaluation runs in total. The script records predictive metrics and runtime for each run.

## Setup

This repository uses the Python environment and dependencies from `chap-core`. Place the two repositories in the same parent directory and set up the `chap-core` environment.

## Usage

From the root of this repository, preview the planned runs:

```bash
uv run --project ../chap-core ./scripts/final_analysis.py --plan
```

Run all experiments:

```bash
uv run --project ../chap-core ./scripts/final_analysis.py
```

Run only one model:

```bash
uv run --project ../chap-core ./scripts/final_analysis.py --model-name mstl_multistep_model
```

## Repository structure

- `scripts/` — Experiment scripts
- `search_spaces/` — HPO search-space definitions
- `model_configurations/` — Model configuration files
- `results/` — Generated evaluation results

## Outputs

Results are written to `results/final_analysis/`, including:

- `summary_all.csv` (or `summary_<model>.csv`) — Evaluation metrics, runtimes, and errors
- `.nc` files — Individual evaluation results
- `log_all.txt` (or `log_<model>.txt`) — Execution logs
- HPO leaderboard CSV files for trials within successful HPO runs
