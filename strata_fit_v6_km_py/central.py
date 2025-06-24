import pandas as pd
from typing import Dict, List, Union, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from strata_fit_v6_km_py.preprocessing import compute_d2t_prevalence_by_year

from .types import (
    NoiseType,
    DEFAULT_INTERVAL_START_COLUMN,
    DEFAULT_CUMULATIVE_INCIDENCE_COLUMN,
    MINIMUM_ORGANIZATIONS
)

@algorithm_client
def kaplan_meier_central(
    client: AlgorithmClient,
    organizations_to_include: Optional[List[int]] = None,
    noise_type: NoiseType = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Central orchestration of the federated Kaplan-Meier algorithm with interval censoring.
    This function uses hyperparameters for column names defined in types.py and calls the preprocessing
    functions automatically before executing the partial tasks.
    
    Returns
    -------
    dict
        The aggregated Kaplan-Meier event table as a JSON table with columns:
      - interval_start
      - removed, observed, interval, censored, at_risk, hazard
      - cumulative_incidence
    """
    if not organizations_to_include:
        organizations_to_include = [org["id"] for org in client.organization.list()]

    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS}).")

    info("Step 1: Collecting unique event times.")
    unique_event_times_results = _start_partial_and_collect_results(
        client,
        method="get_unique_event_times",
        organizations_to_include=organizations_to_include,
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
        client,
        method="get_km_event_table",
        organizations_to_include=organizations_to_include,
        unique_event_times=unique_event_times,
        noise_type=noise_type,
        snr=snr,
        random_seed=random_seed,
    )
    local_event_tables = [pd.read_json(result) for result in local_event_tables_results]

    info("Step 3: Aggregating local event tables.")
    km_df = pd.concat(local_event_tables).groupby(DEFAULT_INTERVAL_START_COLUMN, as_index=False).sum()
    km_df["hazard"] = (km_df["observed"] + km_df["interval"] * 0.5) / km_df["at_risk"]
    km_df[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN] = 1 - (1 - km_df["hazard"]).cumprod()

#4 Collect raw patient-level data for prevalence computation
    info("Step 4: Collecting raw patient-level data for D2T prevalence.")
    local_raw_dfs = _start_partial_and_collect_results(
        client,
        method="get_raw_patient_data",  # ✅ You must ensure this partial exists!
        organizations_to_include=organizations_to_include,
    )
    raw_patient_dfs = [pd.read_json(result) for result in local_raw_dfs]
    combined_df = pd.concat(raw_patient_dfs, ignore_index=True)

    info("Computing D2T-RA prevalence by year.")
    prevalence_df = compute_d2t_prevalence_by_year(combined_df)

    info("Kaplan-Meier curve with interval censoring computed.")
    return {
    "km_result": km_df.to_json(),
    "d2t_prevalence": prevalence_df.to_json()
}

def _start_partial_and_collect_results(
    client: AlgorithmClient,
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
