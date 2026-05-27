from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_unique_dmards(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    ordered["tsDMARD_binary"] = ordered["tsDMARD"].apply(
        lambda value: np.nan if pd.isna(value) else (1 if float(value) > 0 else 0)
    )

    def unique_classes(group: pd.DataFrame) -> pd.Series:
        seen: set[tuple[str, int]] = set()
        counts: list[int] = []
        for bdmard, tsdmard in zip(group["bDMARD"], group["tsDMARD_binary"]):
            if pd.notna(bdmard) and float(bdmard) > 0:
                seen.add(("b", int(bdmard)))
            if pd.notna(tsdmard) and float(tsdmard) > 0:
                seen.add(("t", 1))
            counts.append(len(seen))
        return pd.Series(counts, index=group.index)

    return ordered.groupby("pat_ID", group_keys=False).apply(unique_classes)


def _tag_d2t_like(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["pat_ID", "Visit_months_from_diagnosis"]).copy()
    ordered["cum_unique_btsDMARD"] = _compute_unique_dmards(ordered)
    ordered["last_dmard_change_id"] = (
        ordered.groupby("pat_ID")["cum_unique_btsDMARD"].transform(lambda values: values.ne(values.shift()).cumsum())
    )
    ordered["last_dmard_start_month"] = (
        ordered.groupby(["pat_ID", "last_dmard_change_id"])["Visit_months_from_diagnosis"].transform("min")
    )
    ordered["months_since_last_dmard"] = ordered["Visit_months_from_diagnosis"] - ordered["last_dmard_start_month"]
    ordered["rolling_avg_DAS28"] = (
        ordered.groupby("pat_ID")["DAS28"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    ordered["rolling_avg_CRP"] = (
        ordered.groupby("pat_ID")["CRP"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    ordered["D2T_crit1"] = (ordered["cum_unique_btsDMARD"] >= 2) & (ordered["months_since_last_dmard"] >= 6)
    ordered["D2T_crit2"] = (ordered["rolling_avg_DAS28"] > 3.2) | (ordered["rolling_avg_CRP"] > 1.0)
    ordered["D2T_crit3"] = (ordered["Pat_global"] > 50) | (ordered["Ph_global"] > 50)
    ordered["D2T_RA"] = ordered["D2T_crit1"] & ordered["D2T_crit2"] & ordered["D2T_crit3"]
    return ordered


def _patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    tagged = _tag_d2t_like(df)
    rows: list[dict[str, Any]] = []
    for patient_id, group in tagged.groupby("pat_ID"):
        rows.append(
            {
                "pat_ID": patient_id,
                "sex": group["Sex"].dropna().iloc[0] if not group["Sex"].dropna().empty else None,
                "rf_positivity": group["RF_positivity"].dropna().iloc[0] if not group["RF_positivity"].dropna().empty else None,
                "anti_ccp": group["anti_CCP"].dropna().iloc[0] if not group["anti_CCP"].dropna().empty else None,
                "ever_csDMARD": bool(
                    (
                        pd.concat([group[column] for column in ("csDMARD1", "csDMARD2", "csDMARD3")], axis=0)
                        .pipe(_safe_numeric)
                        .fillna(0)
                        > 0
                    ).any()
                ),
                "ever_bDMARD": bool((_safe_numeric(group["bDMARD"]).fillna(0) > 0).any()),
                "ever_tsDMARD": bool((_safe_numeric(group["tsDMARD"]).fillna(0) > 0).any()),
                "ever_GC": bool((_safe_numeric(group["GC"]).fillna(0) > 0).any()),
                "d2t_like_ever": bool(group["D2T_RA"].fillna(False).any()),
            }
        )
    return pd.DataFrame(rows)


def filter_dataframe_for_cohort(df: pd.DataFrame, cohort: dict[str, Any] | None) -> pd.DataFrame:
    if not cohort:
        return df
    patient_df = _patient_summary(df)
    filtered = patient_df.copy()
    population = cohort.get("population", "all_ra")
    if population == "d2t_like":
        filtered = filtered[filtered["d2t_like_ever"]]
    elif population == "non_d2t_like":
        filtered = filtered[~filtered["d2t_like_ever"]]

    sex = cohort.get("sex", "any")
    if sex == "male":
        filtered = filtered[filtered["sex"] == 0]
    elif sex == "female":
        filtered = filtered[filtered["sex"] == 1]

    rf = cohort.get("rf_positivity", "any")
    if rf == "positive":
        filtered = filtered[filtered["rf_positivity"] == 1]
    elif rf == "negative":
        filtered = filtered[filtered["rf_positivity"] == 0]

    anti_ccp = cohort.get("anti_ccp", "any")
    if anti_ccp == "positive":
        filtered = filtered[filtered["anti_ccp"] == 1]
    elif anti_ccp == "negative":
        filtered = filtered[filtered["anti_ccp"] == 0]

    drug_class = cohort.get("drug_class", "any")
    if drug_class == "csDMARD":
        filtered = filtered[filtered["ever_csDMARD"]]
    elif drug_class == "bDMARD":
        filtered = filtered[filtered["ever_bDMARD"]]
    elif drug_class == "tsDMARD":
        filtered = filtered[filtered["ever_tsDMARD"]]
    elif drug_class == "GC":
        filtered = filtered[filtered["ever_GC"]]

    drug_code = cohort.get("drug_code")
    if drug_code is not None and drug_class != "any":
        keep_ids: list[str] = []
        for patient_id, group in df.groupby("pat_ID"):
            if patient_id not in set(filtered["pat_ID"].tolist()):
                continue
            if drug_class == "csDMARD":
                matched = any((_safe_numeric(group[column]) == int(drug_code)).fillna(False).any() for column in ("csDMARD1", "csDMARD2", "csDMARD3"))
            elif drug_class == "bDMARD":
                matched = ((_safe_numeric(group["bDMARD"]) == int(drug_code)).fillna(False)).any()
            elif drug_class == "tsDMARD":
                matched = ((_safe_numeric(group["tsDMARD"]) == int(drug_code)).fillna(False)).any()
            elif drug_class == "GC" and "GC_type" in group.columns:
                matched = ((_safe_numeric(group["GC_type"]) == int(drug_code)).fillna(False)).any()
            else:
                matched = False
            if matched:
                keep_ids.append(patient_id)
        filtered = filtered[filtered["pat_ID"].isin(keep_ids)]

    return df[df["pat_ID"].isin(filtered["pat_ID"])].copy()
