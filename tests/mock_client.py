import warnings
import os
import pandas as pd
from pathlib import Path

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

warnings.filterwarnings("ignore")

from strata_fit_v6_km_py.types import DEFAULT_INTERVAL_START_COLUMN, DEFAULT_CUMULATIVE_INCIDENCE_COLUMN

def plot_km_curve(df_km):
    import matplotlib.pyplot as plt
    # convert months → years
    years = df_km["interval_start"] / 12
    cum_inc = df_km["cumulative_incidence"]

    plt.figure(figsize=(8, 5))
    plt.step(years, cum_inc, where='post', lw=2)
    plt.xlabel("Years from diagnosis")
    plt.ylabel("Cumulative incidence of D2T-RA")
    plt.title("Cumulative incidence of difficult-to-treat RA (KM estimate)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _fmt_percentage(value):
    return f"{value:.1f}%" if pd.notna(value) else "NA"


def _fmt_mean_sd(mean_value, sd_value):
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(sd_value):
        return f"{mean_value:.2f}"
    return f"{mean_value:.2f} ({sd_value:.2f})"


def build_d2t_characteristics_display_table(df_char):
    row = df_char.iloc[0]
    return pd.DataFrame(
        {
            "Characteristic": [
                "D2T patients, n",
                "Female, %",
                "RF positive, %",
                "Anti-CCP positive, %",
                "Age at diagnosis, mean (SD)",
                "DAS28 at D2T classification, mean (SD)",
            ],
            "Value": [
                int(row["d2t_patients"]),
                _fmt_percentage(row["female_percentage"]),
                _fmt_percentage(row["rf_positive_percentage"]),
                _fmt_percentage(row["anti_ccp_positive_percentage"]),
                _fmt_mean_sd(row["age_mean"], row["age_sd"]),
                _fmt_mean_sd(row["das28_mean_at_d2t"], row["das28_sd_at_d2t"]),
            ],
        }
    )

# --- 1. Define the per-node datasets ---
# Use the same real CSV for each mock organization so the federated pipeline
# can still run with the minimum 3 organizations in the mock client.
source_file = Path("/data/mock_data/strata_fit_v6_km_py/mock_data.csv")  # adjust as needed
dataset1 = {"database": source_file, "db_type": "csv"}
dataset2 = {"database": source_file, "db_type": "csv"}
dataset3 = {"database": source_file, "db_type": "csv"}

# We have three “organizations” in this mock run:
org_ids = [0, 1, 2]

# --- 2. Instantiate the mock client with our module name ---
#    Make sure `module` here matches the name in your setup.py (i.e. the package name).
client = MockAlgorithmClient(
    datasets=[[dataset1], [dataset2], [dataset3]],
    organization_ids=org_ids,
    module="strata_fit_v6_km_py"
)

# --- 3. Trigger the central orchestration ---
# Only send the “master” task to one org; the central function will fan out
# to all three under the hood.
task = client.task.create(
    input_={
        "method": "kaplan_meier_central",
        "kwargs": {
            'organizations_to_include': [0,1,2]
            # you can override noise parameters here if you like,
            # e.g. "noise_type": "GAUSSIAN", "snr": 10, "random_seed": 42
        }
    },
    organizations=[org_ids[0]]
)

# --- 4. Collect and parse the result ---
results_json = client.result.get(task["id"])
df_km = pd.read_json(results_json["km_result"])
df_prev = pd.read_json(results_json["d2t_prevalence"])
df_char = pd.read_json(results_json["d2t_characteristics"])
df_char_display = build_d2t_characteristics_display_table(df_char)


# --- 5. Inspect / assert ---
print("Kaplan–Meier curve (first 5 rows):")
print(df_km.head(), "\n")

print("Summary statistics:")
# print(df_km[["at_risk", "observed", "censored", "interval", "hazard", DEFAULT_CUMULATIVE_INCIDENCE_COLUMN]].describe())
print(df_km.describe())

print("\n📊 D2T-RA Prevalence by Calendar Year:")
print(df_prev)

print("\nD2T population characteristics:")
print(df_char_display.to_string(index=False))


# Example assertion (ensure we have at least one time‐point and survival_cdf is ≤1):
assert not df_km.empty
assert df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN].max() <= 1.0

# plotting
plot_km_curve(df_km)

print("\n✅ Central Kaplan–Meier test completed successfully.")
