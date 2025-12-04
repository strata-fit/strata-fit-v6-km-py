import pandas as pd
from typing import Dict, List, Union, Optional

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from .preprocessing import compute_d2t_prevalence_by_year 

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
    Orchestrates the federated Kaplan-Meier survival analysis with interval censoring
    across multiple organizations using Vantage6.

    This central function coordinates the following steps:
    1. **Unique Event Time Collection**: Triggers the `get_unique_event_times` task on each
       participating organization to extract all unique event timepoints from local data
       after preprocessing and optional noise injection.
    
    2. **Event Table Generation**: Triggers the `get_km_event_table` task to compute local
       Kaplan-Meier event tables (with interval, exact, and censored events) using the
       unified list of event times.

    3. **Aggregation**: Combines local event tables into a single federated table,
       computes hazard rates and cumulative incidence using interval-censoring logic.

    4. **D2T-RA Prevalence Analysis**: Collects raw patient-level visit data from all
       organizations via the `get_raw_patient_data` task and computes year-wise prevalence
       of Difficult-to-Treat Rheumatoid Arthritis (D2T-RA).

    Parameters
    ----------
    client : AlgorithmClient
        Vantage6 client object injected via the `@algorithm_client` decorator.
    
    organizations_to_include : Optional[List[int]], default=None
        List of organization IDs to include in the computation. If not provided,
        all organizations in the collaboration will be used.
    
    noise_type : NoiseType, default=NoiseType.NONE
        Type of noise to inject into event times for differential privacy. Options include
        'NONE', 'GAUSSIAN', or 'POISSON'.
    
    snr : Optional[float], default=None
        Signal-to-noise ratio used when applying Gaussian noise. Required if noise_type is 'GAUSSIAN'.
    
    random_seed : Optional[int], default=None
        Seed for random number generation to ensure reproducibility of noise injection.

    Returns
    -------
    dict
        Dictionary containing:
        - "km_result" (str): JSON-encoded DataFrame with columns:
              - interval_start
              - removed, observed, interval, censored, at_risk, hazard
              - cumulative_incidence
        - "d2t_prevalence" (str): JSON-encoded DataFrame of D2T-RA prevalence per year.
    
    Raises
    ------
    PrivacyThresholdViolation
        If the number of organizations included is less than the minimum threshold required for privacy.
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

#4 Collect  data for prevalence computation
    info("Step 4: Collecting D2T-RA prevalence tables from nodes.")
    local_prevalence_results = _start_partial_and_collect_results(
    client,
    method="get_d2t_prevalence_by_year",
    organizations_to_include=organizations_to_include,
    )
    local_prevalence_dfs = [pd.read_json(result) for result in local_prevalence_results]
    prevalence_df = pd.concat(local_prevalence_dfs).groupby("Year_visit", as_index=False).sum()
    prevalence_df["D2T_RA_prevalence"] = (
        prevalence_df["d2t_positive"] / prevalence_df["total_patients"]
    )
   
   
    
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