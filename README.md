# strata-fit-v6-km-py

Standalone STRATA-FIT Kaplan-Meier federation package using:

- `python:3.11-slim`
- embedded `run_context` dispatch
- a lightweight proxy client for central fan-out

The package no longer depends on Harbor `algorithm-base` images or the old
wrapper package.

## Runtime

Container entrypoint:

```bash
python -m strata_fit_v6_km_py.container
```

The runtime expects `RUN_CONTEXT_FILE` to point at a run-context JSON payload.

Supported run-context entrypoints:

- `kaplan_meier_central`
- `get_unique_event_times`
- `get_km_event_table`
- `get_d2t_prevalence_by_year`

## Install

```bash
pip install -e .[dev]
```

## Local multi-dataset run

Run the pure-Python local runner against three or more CSV partitions:

```bash
python tests/mock_client.py \
  --dataset tests/data/data_times/alpha.csv \
  --dataset tests/data/data_times/beta.csv \
  --dataset tests/data/data_times/gamma.csv
```

The runner prints:

- KM result preview
- summary statistics
- yearly D2T prevalence

Optional plot output:

```bash
python tests/mock_client.py \
  --dataset-glob "tests/data/data_times/*.csv" \
  --plot
```

## Federated Vantage6 run

Use the manual submission helper for a real Vantage6 deployment:

```bash
export V6_SERVER_HOST=https://example.org
export V6_SERVER_PORT=443
export V6_API_PATH=/api
export V6_USERNAME=my-user
export V6_PASSWORD=my-password
export V6_COLLABORATION_ID=3
export V6_MASTER_ORG_ID=5
export V6_ORGANIZATION_IDS=5,6,7
export V6_DATASET_LABEL=dataset_202504
export V6_ALGO_IMAGE=ghcr.io/strata-fit/strata-fit-v6-km-py:release-v1

python tests/federated_run.py
```

The helper submits `kaplan_meier_central` and prints the decoded result.

## Docker build

```bash
docker build -t strata-fit-v6-km-py:local .
```

## Tests

```bash
pytest
```
