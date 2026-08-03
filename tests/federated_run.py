from getpass import getpass
from vantage6.client import Client
from time import sleep
import json
import pandas as pd
import pprint as pprint
from io import StringIO



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

# CHANGED: Combined your config into a simple dict without Pydantic, Dynaconf, or validators
config = {
    'server_url': "https://stratafit.prod.medicaldataworks.nl",   
    'server_port': 443,
    'server_api': "/api",
    'username': "chiara-umcutrecht",                                  
    'password': getpass("Password: "),
    'mfa_code': getpass("2FA: "),                     
    'organization_key': r"/Users/cripepi2/Desktop/Coding/privkey_UMCUtrecht.pem"                               # Optional for encryption
}

# CHANGED: Initialize and authenticate client
client = Client(config['server_url'], config['server_port'], config['server_api'])
client.authenticate(config['username'], config['password'], mfa_code=config['mfa_code'])

if config['organization_key']:
    client.setup_encryption(config['organization_key'])    # 🔴 Optional encryption

# 🔴 OPTIONAL: List organizations and collaborations
print("Available collaborations:")
print(client.collaboration.list(fields=['id', 'name']))
print("\nAvailable organizations:")
print(client.organization.list(fields=['id', 'name']))

# 🔴 CHANGED: Define task input (you can replace this with your desired algorithm config)
task_input = {
    'method': 'kaplan_meier_central',                             # 🔴 Example method
    'kwargs': {
        'organizations_to_include': [5, 11, 8 ],               # IDs of orgs to include
        'noise_type': "GAUSSIAN",
        'snr': 200,
        'random_seed': 2025
    }
}

# 🔴 CHANGED: Define task payload
task = client.task.create(
    collaboration=3,
    organizations=[5],
    name="demo-stats-task",
    image="ghcr.io/strata-fit/strata-fit-v6-km-py@sha256:67e5b5512f7eb8b99f592e32303884d32e2ea56a5b7c270f8f8f62a672b6b59a",
    description="KM",
    databases=[{'label': 'dataset_202504'}],
    input_=task_input
)

# 🔴 CHANGED: Wait for results
print("\nWaiting for results...")
task_id = task["id"]
result_info = client.wait_for_results(task_id)
result_data = client.result.from_task(task_id=task_id)

# Parse central result (double-encoded JSON)
raw_result = result_data["data"][0]["result"]     # str
outer = json.loads(raw_result)

# KM table
df_km = pd.read_json(StringIO(outer["km_result"]))

# Prevalence table
df_prev = pd.read_json(StringIO(outer["d2t_prevalence"]))

# D2T characteristics table
df_char = pd.read_json(StringIO(outer["d2t_characteristics"]))
df_char_display = build_d2t_characteristics_display_table(df_char)

print("\nD2T-RA prevalence table (first 20 rows):")
print(df_prev.head(20), "\n")

print("D2T population characteristics:")
print(df_char_display.to_string(index=False), "\n")

# --- Inspect / assert ---
print("Kaplan–Meier curve (first 5 rows):")
print(df_km.head(), "\n")

print("Summary statistics:")
print(df_km.describe())

assert not df_km.empty
assert df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN].max() <= 1.0

# Plot
plot_km_curve(df_km)

print("\n✅ Central Kaplan-Meier test completed successfully.")
