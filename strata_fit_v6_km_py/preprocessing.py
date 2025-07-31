"""
Preprocessing functions to transform raw STRATA-FIT data
into standardized interval survival data for federated KM.
"""

import pandas as pd
import numpy as np
from .types import (
    EventType, 
    DEFAULT_INTERVAL_START_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_EVENT_INDICATOR_COLUMN
)

def compute_unique_dmards(df):
    """
    Compute the cumulative count of unique bDMARD and tsDMARD drug classes used 
    per patient over time.

    The function assumes that `bDMARD` and `tsDMARD` columns contain integer-encoded
    drug class identifiers (rather than one-hot or categorical format), consistent
    with the schema defined in:
    https://github.com/mdw-nl/strata-fit-data-schema/commit/056faf101ccd12ea555986ef6b0b1f0df90db2ce

    For each patient (`pat_ID`), the function iterates through all visits sorted 
    by `Visit_months_from_diagnosis` and tracks how many distinct drug classes 
    (across both bDMARD and tsDMARD) have been used up to each timepoint.

    This transformation is necessary to correctly identify transitions in 
    therapeutic strategy (i.e., changes in mechanism of action), which are 
    required for downstream time-to-event analyses such as the Kaplan-Meier 
    algorithm used in STRATA-FIT.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns `pat_ID`, 
                           `Visit_months_from_diagnosis`, `bDMARD`, `tsDMARD`.

    Returns:
        pd.Series: A Series with the cumulative number of unique DMARD 
                   classes per visit, indexed like the input DataFrame.
    """
    df = df.sort_values(['pat_ID', 'Visit_months_from_diagnosis']).copy()

    def unique_classes(sub_df):
        unique_b = []
        unique_ts = []
        counts = []

        for b, t in zip(sub_df['bDMARD'], sub_df['tsDMARD']):
            if not pd.isna(b) and b not in unique_b:
                unique_b.append(b)
            if not pd.isna(t) and t not in unique_ts:
                unique_ts.append(t)
            total_unique = len(set(unique_b + unique_ts))
            counts.append(total_unique)

        return pd.Series(counts, index=sub_df.index)

    return df.groupby('pat_ID', group_keys=False).apply(unique_classes)

def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess STRATA-FIT input data into interval survival format suitable 
    for federated Kaplan-Meier analysis.

    This function applies multiple transformations:
    1. **Clipping diagnosis year**: Patients diagnosed before 2006 are shifted to 2006. 
       Corresponding visit months are adjusted to preserve visit calendar dates.
       Visits occurring before this clipped diagnosis time are removed.

    2. **DMARD exposure tracking**: Calculates the cumulative number of unique 
       bDMARD and tsDMARD drug classes used per patient. This is used to 
       identify those who meet EULAR D2T RA criterion 1.

    3. **Disease activity metrics**: Computes a rolling average of DAS28 to 
       help assess sustained disease activity.

    4. **D2T RA classification**: Applies simplified operational definitions for 
       the three EULAR D2T RA criteria using available schema variables:
         - Criterion 1: ≥2 distinct b/tsDMARD classes
         - Criterion 2: DAS28 > 3.2 or rolling DAS28 > 3.2
         - Criterion 3: Pat_global or Ph_global > 50
       A patient is considered D2T RA if all three criteria are met at any visit.

    5. **Per-patient aggregation**: Determines time-to-event (TTE), censoring status,
       and constructs interval survival data fields:
         - `interval_start`: 0 if interval-censored, otherwise TTE
         - `interval_end`: min follow-up if interval-censored, otherwise TTE
         - `event_indicator`: 'interval', 'exact', or 'censored' based on availability and criteria

    Parameters:
        df (pd.DataFrame): Raw STRATA-FIT input DataFrame containing variables such as 
                           `pat_ID`, `Visit_months_from_diagnosis`, `bDMARD`, `tsDMARD`, 
                           `DAS28`, `Pat_global`, `Ph_global`, `Year_diagnosis`, etc.

    Returns:
        pd.DataFrame: A summarized DataFrame (one row per patient) with:
            - `interval_start`, `interval_end`, `event_indicator`
            - `TTE`, `maxFU`, `minFU`, `D2T_RA_Ever`, `cens`
            - Fields necessary for interval-censored survival modeling
    """
    # Sort data by patient ID and follow-up time and remove time before 2006
    df.sort_values(['pat_ID', 'Visit_months_from_diagnosis'], inplace=True)
    shift_mask = df['Year_diagnosis'] < 2006
    year_shift = 2006 - df.loc[shift_mask, 'Year_diagnosis']
    df.loc[shift_mask, 'Visit_months_from_diagnosis'] = (
        df.loc[shift_mask, 'Visit_months_from_diagnosis'] - year_shift * 12
    )
    df.loc[shift_mask, 'Year_diagnosis'] = 2006
    df = df[df['Visit_months_from_diagnosis'] >= 0].reset_index(drop=True)

    # Compute cumulative unique DMARD classes
    df['cum_unique_btsDMARD'] = compute_unique_dmards(df)
    df['cum_btsDMARDmin'] = df.groupby('pat_ID')['cum_unique_btsDMARD'].cummin()
    

    # Rolling average DAS28 (optional improvement)
    df['rolling_avg_DAS28'] = df.groupby('pat_ID')['DAS28'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Step 2: Define criteria for D2T RA
    df['D2T_crit1'] = df['cum_unique_btsDMARD'] >= 2
    df['D2T_crit2'] = (df['DAS28'] > 3.2) | (df['rolling_avg_DAS28'] > 3.2)
    df['D2T_crit3'] = (df['Pat_global'] > 50) | (df['Ph_global'] > 50)

    df['D2T_RA'] = df['D2T_crit1'] & df['D2T_crit2'] & df['D2T_crit3']

    # Step 3: Per-patient summary
    summary = df.groupby('pat_ID').agg(
        Year_diagnosis=('Year_diagnosis', 'first'),
        D2T_RA_Ever=('D2T_RA', 'max'),
        cum_btsDMARDmin=('cum_btsDMARDmin', 'max'),
        minFU=('Visit_months_from_diagnosis', 'min'),
        TTE=('Visit_months_from_diagnosis', lambda x: x[df.loc[x.index, 'D2T_RA']].min() if any(df.loc[x.index, 'D2T_RA']) else np.nan),
        maxFU=('Visit_months_from_diagnosis', 'max')
    ).reset_index()

    summary['D2T_RA_Ever'] = summary['D2T_RA_Ever'].fillna(0)
    summary['cum_btsDMARDmin'] = summary['cum_btsDMARDmin'].fillna(0)
    summary['TTE'] = summary['TTE'].fillna(summary['maxFU'])

    # Step 4: Define censoring type
    summary['cens'] = np.select(
        condlist=[
            (summary['D2T_RA_Ever'] == 1) & (summary['cum_btsDMARDmin'] > 2),
            (summary['D2T_RA_Ever'] == 0)
        ],
        choicelist=['interval', 'right'],
        default='no'
    )

    # Step 5: Define interval start, interval end, and event type
    summary[DEFAULT_INTERVAL_START_COLUMN] = np.where(summary['cens'] == 'interval', 0, summary['TTE'])
    summary[DEFAULT_INTERVAL_END_COLUMN] = np.where(summary['cens'] == 'interval', summary['minFU'], summary['TTE'])
    summary[DEFAULT_EVENT_INDICATOR_COLUMN] = np.select(
        condlist=[
            (summary['cens'] == 'interval'),
            (summary['cens'] == 'no')
        ],
        choicelist=[EventType.INTERVAL.value, EventType.EXACT.value],
        default=EventType.CENSORED.value
    )

    return summary

def compute_d2t_prevalence_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes D2T prevalence by calendar year using minimal data.

    Returns only yearly aggregates: total patients and D2T-positive counts.
    """
    df = df.sort_values(['pat_ID', 'Visit_months_from_diagnosis']).copy()

    shift_mask = df['Year_diagnosis'] < 2006
    year_shift = 2006 - df.loc[shift_mask, 'Year_diagnosis']
    df.loc[shift_mask, 'Visit_months_from_diagnosis'] -= year_shift * 12
    df.loc[shift_mask, 'Year_diagnosis'] = 2006
    df = df[df['Visit_months_from_diagnosis'] >= 0].reset_index(drop=True)

    df['cum_unique_btsDMARD'] = compute_unique_dmards(df)
    df['cum_btsDMARDmin'] = df.groupby('pat_ID')['cum_unique_btsDMARD'].cummin()
    df['rolling_avg_DAS28'] = df.groupby('pat_ID')['DAS28'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    df['D2T_crit1'] = df['cum_unique_btsDMARD'] >= 2
    df['D2T_crit2'] = (df['DAS28'] > 3.2) | (df['rolling_avg_DAS28'] > 3.2)
    df['D2T_crit3'] = (df['Pat_global'] > 50) | (df['Ph_global'] > 50)
    df['D2T_RA'] = df['D2T_crit1'] & df['D2T_crit2'] & df['D2T_crit3']

    df["Year_visit"] = df["Year_diagnosis"] + (df["Visit_months_from_diagnosis"] / 12).astype(int)
    df["d2t_positive"] = df["D2T_RA"]

    d2t_by_year = (
        df.groupby("Year_visit")
        .agg(
            total_patients=("pat_ID", "nunique"),
            d2t_positive=("d2t_positive", "sum")
        )
        .reset_index()
    )

    return d2t_by_year
