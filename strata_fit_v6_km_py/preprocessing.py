"""
Shared preprocessing functions for transforming raw STRATA-FIT data into
derived D2T features and interval-censored KM inputs.
"""

import numpy as np
import pandas as pd

from .types import (
    DEFAULT_D2T_STEP3_COLUMN,
    DEFAULT_EVENT_INDICATOR_COLUMN,
    DEFAULT_EVER_D2T_COLUMN,
    DEFAULT_FIRST_D2T_MONTH_COLUMN,
    DEFAULT_INTERVAL_END_COLUMN,
    DEFAULT_INTERVAL_START_COLUMN,
    EventType,
)


def compute_unique_dmards(df: pd.DataFrame) -> pd.Series:
    """
    Compute the cumulative count of unique bDMARD and tsDMARD classes per visit.

    bDMARD class identifiers are counted as distinct classes as-is. tsDMARD is
    collapsed to a single exposure class: any non-zero tsDMARD value counts as
    one shared "tsDMARD exposed" class.
    """
    df = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()

    df["tsDMARD_binary"] = df["tsDMARD"].apply(
        lambda x: np.nan if pd.isna(x) else (1 if x != 0 else 0)
    )

    def unique_classes(sub_df: pd.DataFrame) -> pd.Series:
        seen = set()
        counts = []

        for bdmard_class, tsdmard_exposed in zip(
            sub_df["bDMARD"], sub_df["tsDMARD_binary"]
        ):
            if not pd.isna(bdmard_class):
                seen.add(("b", bdmard_class))
            if tsdmard_exposed == 1:
                seen.add(("t", 1))
            counts.append(len(seen))

        return pd.Series(counts, index=sub_df.index)

    return df.groupby("pat_ID", group_keys=False).apply(unique_classes)


def normalize_strata_fit_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort visits and clip diagnosis year to 2006 while preserving calendar dates.
    """
    visits = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()

    shift_mask = visits["Year_diagnosis"] < 2006
    year_shift = 2006 - visits.loc[shift_mask, "Year_diagnosis"]
    visits.loc[shift_mask, "Visit_months_from_diagnosis"] = (
        visits.loc[shift_mask, "Visit_months_from_diagnosis"] - year_shift * 12
    )
    visits.loc[shift_mask, "Year_diagnosis"] = 2006

    return visits[visits["Visit_months_from_diagnosis"] >= 0].reset_index(drop=True)


def derive_d2t_visit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build reusable visit-level D2T features from raw STRATA-FIT visit data.

    Returns the visit table with explicit shared derived fields, including:
    - `ever_d2t` via patient aggregation helper
    - `first_d2t_month` via patient aggregation helper
    - `D2T_step3` as the visit-level final D2T flag
    """
    visits = normalize_strata_fit_visits(df)

    visits["cum_unique_btsDMARD"] = compute_unique_dmards(visits)
    visits["cum_btsDMARDmin"] = visits.groupby("pat_ID")["cum_unique_btsDMARD"].cummin()
    visits["last_dmard_change_id"] = (
        visits.groupby("pat_ID")["cum_unique_btsDMARD"]
        .transform(lambda x: x.ne(x.shift()).cumsum())
    )
    visits["last_dmard_start_month"] = (
        visits.groupby(["pat_ID", "last_dmard_change_id"])["Visit_months_from_diagnosis"]
        .transform("min")
    )
    visits["months_since_last_dmard"] = (
        visits["Visit_months_from_diagnosis"] - visits["last_dmard_start_month"]
    )

    visits["rolling_avg_DAS28"] = (
        visits.groupby("pat_ID")["DAS28"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    visits["rolling_avg_CRP"] = (
        visits.groupby("pat_ID")["CRP"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    visits["D2T_crit1"] = (
        (visits["cum_unique_btsDMARD"] >= 2)
        & (visits["months_since_last_dmard"] >= 6)
    )
    visits["D2T_crit2"] = (
        (visits["rolling_avg_DAS28"] > 3.2) | (visits["rolling_avg_CRP"] > 1.0)
    )
    visits["D2T_crit3"] = (visits["Pat_global"] > 50) | (visits["Ph_global"] > 50)
    visits[DEFAULT_D2T_STEP3_COLUMN] = (
        visits["D2T_crit1"] & visits["D2T_crit2"] & visits["D2T_crit3"]
    )

    # Preserve the legacy visit-level flag name for downstream compatibility.
    visits["D2T_RA"] = visits[DEFAULT_D2T_STEP3_COLUMN]
    visits["Year_visit"] = (
        visits["Year_diagnosis"] + (visits["Visit_months_from_diagnosis"] / 12).astype(int)
    )

    return visits


def summarize_patient_level_d2t(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize reusable patient-level D2T derived features.
    """
    visits = derive_d2t_visit_features(df)

    summary = visits.groupby("pat_ID").agg(
        Year_diagnosis=("Year_diagnosis", "first"),
        cum_btsDMARDmin=("cum_btsDMARDmin", "max"),
        minFU=("Visit_months_from_diagnosis", "min"),
        maxFU=("Visit_months_from_diagnosis", "max"),
        ever_d2t=(DEFAULT_D2T_STEP3_COLUMN, "max"),
        first_d2t_month=(
            "Visit_months_from_diagnosis",
            lambda x: x[visits.loc[x.index, DEFAULT_D2T_STEP3_COLUMN]].min()
            if visits.loc[x.index, DEFAULT_D2T_STEP3_COLUMN].any()
            else np.nan,
        ),
    ).reset_index()

    summary[DEFAULT_EVER_D2T_COLUMN] = summary["ever_d2t"].fillna(0).astype(int)
    summary[DEFAULT_FIRST_D2T_MONTH_COLUMN] = summary["first_d2t_month"]

    # Preserve the legacy column names used by the KM pipeline.
    summary["D2T_RA_Ever"] = summary[DEFAULT_EVER_D2T_COLUMN]
    summary["TTE"] = summary[DEFAULT_FIRST_D2T_MONTH_COLUMN].fillna(summary["maxFU"])

    return summary


def strata_fit_data_to_km_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw STRATA-FIT visit data into patient-level KM input.

    The returned table includes the reusable patient-level D2T fields
    `ever_d2t` and `first_d2t_month` alongside the existing KM columns.
    """
    summary = summarize_patient_level_d2t(df)

    summary["cens"] = np.select(
        condlist=[
            (summary["D2T_RA_Ever"] == 1) & (summary["cum_btsDMARDmin"] > 2),
            (summary["D2T_RA_Ever"] == 0),
        ],
        choicelist=["interval", "right"],
        default="no",
    )

    summary[DEFAULT_INTERVAL_START_COLUMN] = np.where(
        summary["cens"] == "interval", 0, summary["TTE"]
    )
    summary[DEFAULT_INTERVAL_END_COLUMN] = np.where(
        summary["cens"] == "interval", summary["minFU"], summary["TTE"]
    )
    summary[DEFAULT_EVENT_INDICATOR_COLUMN] = np.select(
        condlist=[
            summary["cens"] == "interval",
            summary["cens"] == "no",
        ],
        choicelist=[
            EventType.INTERVAL.value,
            EventType.EXACT.value,
        ],
        default=EventType.CENSORED.value,
    )

    return summary


def compute_d2t_prevalence_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-wise D2T prevalence using the shared visit-level derivation.

    Patients are counted at most once per calendar year in the D2T-positive
    numerator, even if they have multiple D2T-positive visits in that year.
    """
    visits = derive_d2t_visit_features(df)

    patient_year = visits.groupby(["Year_visit", "pat_ID"], as_index=False).agg(
        d2t_positive=(DEFAULT_D2T_STEP3_COLUMN, "max")
    )

    return (
        patient_year.groupby("Year_visit", as_index=False)
        .agg(
            total_patients=("pat_ID", "nunique"),
            d2t_positive=("d2t_positive", "sum"),
        )
        .reset_index(drop=True)
    )


def _to_binary_indicator(series: pd.Series, truthy_values: set[str]) -> pd.Series:
    """
    Convert mixed encoded binary/categorical values to 0/1 with NaN preserved.
    """
    normalized = series.astype("string").str.strip().str.lower()
    indicator = pd.Series(np.nan, index=series.index, dtype="float64")
    indicator.loc[normalized.isin(truthy_values)] = 1.0
    indicator.loc[normalized.isin({"0", "false", "f", "no", "n", "male", "m"})] = 0.0

    numeric = pd.to_numeric(series, errors="coerce")
    indicator.loc[numeric == 1] = 1.0
    indicator.loc[numeric == 0] = 0.0

    return indicator


def compute_d2t_characteristics_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute federated-safe aggregate components for the D2T population summary.

    The D2T population is defined as one row per patient at the first visit where
    the patient is classified as D2T.
    """
    visits = derive_d2t_visit_features(df)
    d2t_visits = visits.loc[visits[DEFAULT_D2T_STEP3_COLUMN]].copy()

    if d2t_visits.empty:
        return pd.DataFrame(
            [
                {
                    "d2t_patients": 0,
                    "female_positive_count": 0,
                    "female_non_missing_count": 0,
                    "rf_positive_count": 0,
                    "rf_non_missing_count": 0,
                    "anti_ccp_positive_count": 0,
                    "anti_ccp_non_missing_count": 0,
                    "age_count": 0,
                    "age_sum": 0.0,
                    "age_sum_sq": 0.0,
                    "das28_count": 0,
                    "das28_sum": 0.0,
                    "das28_sum_sq": 0.0,
                }
            ]
        )

    first_d2t = (
        d2t_visits.sort_values(["pat_ID", "Visit_months_from_diagnosis"])
        .drop_duplicates(subset="pat_ID", keep="first")
        .reset_index(drop=True)
    )

    female = _to_binary_indicator(
        first_d2t["Sex"], {"1", "true", "t", "yes", "y", "female", "f"}
    )
    rf_positive = _to_binary_indicator(
        first_d2t["RF_positivity"], {"1", "true", "t", "yes", "y", "positive", "pos"}
    )
    anti_ccp_positive = _to_binary_indicator(
        first_d2t["anti_CCP"], {"1", "true", "t", "yes", "y", "positive", "pos"}
    )

    age = pd.to_numeric(first_d2t["Age_diagnosis"], errors="coerce")
    das28 = pd.to_numeric(first_d2t["DAS28"], errors="coerce")

    return pd.DataFrame(
        [
            {
                "d2t_patients": int(len(first_d2t)),
                "female_positive_count": int(female.fillna(0).sum()),
                "female_non_missing_count": int(female.notna().sum()),
                "rf_positive_count": int(rf_positive.fillna(0).sum()),
                "rf_non_missing_count": int(rf_positive.notna().sum()),
                "anti_ccp_positive_count": int(anti_ccp_positive.fillna(0).sum()),
                "anti_ccp_non_missing_count": int(anti_ccp_positive.notna().sum()),
                "age_count": int(age.notna().sum()),
                "age_sum": float(age.fillna(0).sum()),
                "age_sum_sq": float((age.fillna(0) ** 2).sum()),
                "das28_count": int(das28.notna().sum()),
                "das28_sum": float(das28.fillna(0).sum()),
                "das28_sum_sq": float((das28.fillna(0) ** 2).sum()),
            }
        ]
    )
