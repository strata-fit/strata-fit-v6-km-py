from io import StringIO
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from pathlib import Path

from .preprocessing import compute_d2t_prevalence_by_year 
from .client import AlgorithmProxyClient
from .exceptions import PrivacyThresholdViolation
from .io import write_output
from .log import info
from .partial import (
    compute_d2t_prevalence_by_year_frame,
    compute_km_event_table_frame,
    compute_unique_event_times_frame,
)
from .runtime import run_context

from .types import (
    NoiseType,
    DEFAULT_INTERVAL_START_COLUMN,
    DEFAULT_CUMULATIVE_INCIDENCE_COLUMN,
    MINIMUM_ORGANIZATIONS
)


def _read_json_frame(result: Any) -> pd.DataFrame:
    if isinstance(result, str):
        return pd.read_json(StringIO(result))
    return pd.read_json(result)


def _resolve_organization_ids(
    client: AlgorithmProxyClient,
    organizations_to_include: Optional[List[int] | int],
) -> List[int]:
    if organizations_to_include is None:
        return [org["id"] for org in client.organization.list()]
    if isinstance(organizations_to_include, int):
        return [organizations_to_include]
    return [int(org_id) for org_id in organizations_to_include]


def _aggregate_kaplan_meier_results(
    local_unique_event_times_results: List[List[float]],
    local_event_tables_results: List[Any],
    local_prevalence_results: List[Any],
    organizations_to_include: List[int],
) -> Dict[str, Any]:
    unique_event_times = set()
    for result in local_unique_event_times_results:
        unique_event_times.update(result)
    unique_event_times = sorted(unique_event_times)

    local_event_tables = [_read_json_frame(result) for result in local_event_tables_results]
    km_df = pd.concat(local_event_tables).groupby(DEFAULT_INTERVAL_START_COLUMN, as_index=False).sum()
    km_df["hazard"] = (km_df["observed"] + km_df["interval"] * 0.5) / km_df["at_risk"]
    km_df[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN] = 1 - (1 - km_df["hazard"]).cumprod()

    local_prevalence_dfs = [_read_json_frame(result) for result in local_prevalence_results]
    prevalence_df = pd.concat(local_prevalence_dfs).groupby("Year_visit", as_index=False).sum()
    prevalence_df["D2T_RA_prevalence"] = (
        prevalence_df["d2t_positive"] / prevalence_df["total_patients"]
    )

    series = [
        {
            "time_months": float(row[DEFAULT_INTERVAL_START_COLUMN]),
            "cumulative_incidence": float(row[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN]),
            "at_risk": int(row["at_risk"]),
            "observed": int(row["observed"]),
            "censored": int(row["censored"]),
            "interval": int(row["interval"]),
        }
        for _, row in km_df.iterrows()
    ]
    prevalence_series = [
        {
            "year": int(row["Year_visit"]),
            "prevalence": float(row["D2T_RA_prevalence"]),
        }
        for _, row in prevalence_df.iterrows()
    ]
    return {
        "series": series,
        "prevalence_series": prevalence_series,
        "metadata": {
            "included_organizations": organizations_to_include,
            "event_definition": "current_d2t_like",
            "time_origin": "diagnosis",
        },
        "km_result": km_df.to_json(),
        "d2t_prevalence": prevalence_df.to_json(),
    }


def run_local_kaplan_meier(
    datasets: List[pd.DataFrame],
    cohort: Optional[dict] = None,
    noise_type: NoiseType | str | None = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    organizations_to_include = list(range(len(datasets)))
    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(
            f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS})."
        )

    info("Step 1: Collecting unique event times.")
    unique_event_times_results = [
        compute_unique_event_times_frame(
            dataset,
            cohort=cohort,
            noise_type=noise_type,
            snr=snr,
            random_seed=random_seed,
        )
        for dataset in datasets
    ]
    unique_event_times = sorted({value for result in unique_event_times_results for value in result})

    info("Step 2: Collecting local event tables.")
    local_event_tables_results = [
        compute_km_event_table_frame(
            dataset,
            unique_event_times=unique_event_times,
            cohort=cohort,
            noise_type=noise_type,
            snr=snr,
            random_seed=random_seed,
        )
        for dataset in datasets
    ]

    info("Step 4: Collecting D2T-RA prevalence tables from nodes.")
    local_prevalence_results = [
        compute_d2t_prevalence_by_year_frame(dataset, cohort=cohort)
        for dataset in datasets
    ]

    info("Kaplan-Meier curve with interval censoring computed.")
    return _aggregate_kaplan_meier_results(
        unique_event_times_results,
        local_event_tables_results,
        local_prevalence_results,
        organizations_to_include,
    )


@run_context(
    output_uris="output_path",
    named_arguments=["organizations_to_include", "cohort", "noise_type", "snr", "random_seed"],
)
def kaplan_meier_central(
    organizations_to_include: Optional[List[int]] = None,
    cohort: Optional[dict] = None,
    noise_type: NoiseType | str | None = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
    output_path: str | Path | None = None,
    client: AlgorithmProxyClient | None = None,
) -> Dict[str, Any]:
    resolved_client = client or AlgorithmProxyClient.from_env()
    organizations_to_include = _resolve_organization_ids(
        resolved_client, organizations_to_include
    )

    if not organizations_to_include:
        organizations_to_include = [org["id"] for org in resolved_client.organization.list()]

    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(
            f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS})."
        )

    info("Step 1: Collecting unique event times.")
    unique_event_times_results = _start_partial_and_collect_results(
        resolved_client,
        method="get_unique_event_times",
        organizations_to_include=organizations_to_include,
        cohort=cohort,
        noise_type=noise_type,
        snr=snr,
        random_seed=random_seed,
    )
    unique_event_times = set()
    for result in unique_event_times_results:
        unique_event_times.update(result)
    unique_event_times = sorted(unique_event_times)

    info("Step 2: Collecting local event tables.")
    local_event_tables_results = _start_partial_and_collect_results(
        resolved_client,
        method="get_km_event_table",
        organizations_to_include=organizations_to_include,
        unique_event_times=unique_event_times,
        cohort=cohort,
        noise_type=noise_type,
        snr=snr,
        random_seed=random_seed,
    )

    info("Step 4: Collecting D2T-RA prevalence tables from nodes.")
    local_prevalence_results = _start_partial_and_collect_results(
        resolved_client,
        method="get_d2t_prevalence_by_year",
        organizations_to_include=organizations_to_include,
        cohort=cohort,
    )

    info("Kaplan-Meier curve with interval censoring computed.")
    result = _aggregate_kaplan_meier_results(
        unique_event_times_results,
        local_event_tables_results,
        local_prevalence_results,
        organizations_to_include,
    )
    write_output(output_path, result)
    return result

def _start_partial_and_collect_results(
    client: AlgorithmProxyClient,
    method: str,
    organizations_to_include: List[int],
    **kwargs,
) -> List[Dict]:
    info(f"Starting partial task '{method}' with {len(organizations_to_include)} organizations.")
    task = client.task.create(
        input_={"method": method, "kwargs": kwargs},
        organizations=organizations_to_include,
    )
    info("Waiting for results...")
    results = client.wait_for_results(task_id=task["id"])
    info(f"Results for '{method}' received.")
    return results
