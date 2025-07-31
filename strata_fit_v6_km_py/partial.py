import pandas as pd
import numpy as np
from typing import List, Optional
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.exceptions import InputError

from .types import (
    NoiseType, EventType,
    DEFAULT_INTERVAL_START_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_EVENT_INDICATOR_COLUMN
)
from .utils import add_noise_to_event_times
from .preprocessing import strata_fit_data_to_km_input
from .preprocessing import compute_d2t_prevalence_by_year

@data(1)
def get_unique_event_times(
    df: pd.DataFrame,
    noise_type: NoiseType = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> List[float]:
    """
    Preprocess the data and collect unique event times from the standardized columns.
    """
    info("Starting get_unique_event_times task.")
    info("Running preprocessing on input data for unique event times.")
    df = strata_fit_data_to_km_input(df)
    info(f"Preprocessing complete. Processed {df.shape[0]} rows.")

    # Apply noise to both time columns.
    info("Adding noise to interval start column.")
    df = add_noise_to_event_times(df, DEFAULT_INTERVAL_START_COLUMN, noise_type, snr, random_seed)
    info("Adding noise to interval end column.")
    df = add_noise_to_event_times(df, DEFAULT_INTERVAL_END_COLUMN, noise_type, snr, random_seed)

    unique_times = pd.concat([
        df[DEFAULT_INTERVAL_START_COLUMN],
        df[DEFAULT_INTERVAL_END_COLUMN]
    ]).dropna().unique()
    info(f"Collected {len(unique_times)} unique event times.")

    return unique_times.tolist()


@data(1)
def get_km_event_table(
    df: pd.DataFrame,
    unique_event_times: List[float],
    noise_type: NoiseType = NoiseType.NONE,
    snr: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> str:
    """
    Preprocess the data and generate an event table for Kaplan-Meier calculation with interval censoring.
    """
    info("Starting get_km_event_table task.")
    info("Running preprocessing on input data for KM event table.")
    df = strata_fit_data_to_km_input(df)
    info(f"Preprocessing complete. Processed {df.shape[0]} rows.")

    info("Adding noise to interval start column for KM event table.")
    df = add_noise_to_event_times(df, DEFAULT_INTERVAL_START_COLUMN, noise_type, snr, random_seed)
    info("Adding noise to interval end column for KM event table.")
    df = add_noise_to_event_times(df, DEFAULT_INTERVAL_END_COLUMN, noise_type, snr, random_seed)

    info("Constructing event table based on unique event times.")
    event_table = (
        pd.DataFrame(index=sorted(unique_event_times))
        .rename_axis(DEFAULT_INTERVAL_START_COLUMN)
        .reset_index()  
    )
    
    # Count exact events
    exact_events = df[df[DEFAULT_EVENT_INDICATOR_COLUMN] == EventType.EXACT.value]
    event_counts = exact_events[DEFAULT_INTERVAL_START_COLUMN].value_counts().reindex(unique_event_times, fill_value=0)
    info(f"Exact events counted: {event_counts.sum()}.")

    # Count right-censored events
    censored_events = df[df[DEFAULT_EVENT_INDICATOR_COLUMN] == EventType.CENSORED.value]
    censored_counts = censored_events[DEFAULT_INTERVAL_START_COLUMN].value_counts().reindex(unique_event_times, fill_value=0)
    info(f"Right-censored events counted: {censored_counts.sum()}.")

    # Count interval-censored events using the interval_end column
    interval_events = df[df[DEFAULT_EVENT_INDICATOR_COLUMN] == EventType.INTERVAL.value]
    interval_counts = interval_events[DEFAULT_INTERVAL_END_COLUMN].value_counts().reindex(unique_event_times, fill_value=0)
    info(f"Interval-censored events counted: {interval_counts.sum()}.")

    # Assemble the event table
    total_removed   = event_counts + censored_counts + interval_counts
    total_observed  = event_counts
    total_interval  = interval_counts
    total_censored  = censored_counts

    event_table["removed"]  = total_removed.values
    event_table["observed"] = total_observed.values
    event_table["interval"] = total_interval.values
    event_table["censored"] = total_censored.values

    # Compute at-risk counts using reverse cumulative sum
    event_table["at_risk"] = event_table["removed"].iloc[::-1].cumsum().iloc[::-1]
    info("Event table constructed successfully with at-risk counts computed.")

    return event_table.to_json()

@data(1)
def get_d2t_prevalence_by_year(df: pd.DataFrame) -> str:
    """
    Partial function to compute per-year D2T prevalence on each node.
    """
    from .preprocessing import compute_d2t_prevalence_by_year

    prevalence_df = compute_d2t_prevalence_by_year(df)
    return prevalence_df.to_json()