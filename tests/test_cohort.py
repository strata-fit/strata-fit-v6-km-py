from __future__ import annotations

import pandas as pd

from strata_fit_v6_km_py.cohort import filter_dataframe_for_cohort


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pat_ID": ["A1", "A1", "B1", "B1", "C1"],
            "Visit_months_from_diagnosis": [0, 12, 0, 9, 0],
            "Age_diagnosis": [45, 45, 60, 60, 52],
            "Sex": [1, 1, 0, 0, 1],
            "RF_positivity": [1, 1, 0, 0, 1],
            "anti_CCP": [1, 1, 0, 0, 1],
            "DAS28": [5.2, 5.0, 2.2, 2.0, 3.7],
            "Pat_global": [72, 74, 30, 25, 62],
            "Ph_global": [70, 72, 20, 18, 55],
            "CRP": [2.3, 2.1, 0.4, 0.5, 1.3],
            "csDMARD1": [1, 1, 2, 2, 1],
            "csDMARD2": [None, None, None, None, None],
            "csDMARD3": [None, None, None, None, None],
            "bDMARD": [1, 3, None, None, 2],
            "tsDMARD": [None, 1, None, None, None],
            "GC": [1, 1, 0, 0, 1],
        }
    )


def test_filter_dataframe_for_bdmard_and_d2t_population() -> None:
    filtered = filter_dataframe_for_cohort(
        _dataset(),
        {
            "population": "d2t_like",
            "drug_class": "bDMARD",
            "drug_code": None,
            "sex": "any",
            "rf_positivity": "any",
            "anti_ccp": "any",
        },
    )
    assert set(filtered["pat_ID"].unique()) == {"A1", "C1"}


def test_filter_dataframe_for_specific_sex_and_rf() -> None:
    filtered = filter_dataframe_for_cohort(
        _dataset(),
        {
            "population": "all_ra",
            "drug_class": "any",
            "drug_code": None,
            "sex": "male",
            "rf_positivity": "negative",
            "anti_ccp": "negative",
        },
    )
    assert set(filtered["pat_ID"].unique()) == {"B1"}
