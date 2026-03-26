from getpass import getpass
from vantage6.client import Client
from time import sleep
import json
import pandas as pd


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
        'organizations_to_include': [5],
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
    image="ghcr.io/mdw-nl/strata-fit-v6-km-py@sha256:bc4d691aac6da06767b813800557c2868da2e1b30121ffeaf0c1211bd9f739a1",
    description="KM",
    databases=[{'label': 'dataset_202504'}],
    input_=task_input
)

# 🔴 CHANGED: Wait for results
print("\nWaiting for results...")
task_id = task["id"]
result_info = client.wait_for_results(task_id)
result_data = client.result.from_task(task_id=task_id)

# 🔴 Display nicely
print("\nResults:")
for item in result_data['data']:
    print(json.dumps(item['result'], indent=2))


df_km = pd.read_json(json.loads(result_data['data'][0]['result']))

# --- 5. Inspect / assert ---
print("Kaplan–Meier curve (first 5 rows):")
print(df_km.head(), "\n")

print("Summary statistics:")
# print(df_km[["at_risk", "observed", "censored", "interval", "hazard", DEFAULT_CUMULATIVE_INCIDENCE_COLUMN]].describe())
print(df_km.describe())

# Example assertion (ensure we have at least one time‐point and survival_cdf is ≤1):
assert not df_km.empty
assert df_km[DEFAULT_CUMULATIVE_INCIDENCE_COLUMN].max() <= 1.0

# plotting
plot_km_curve(df_km)

print("\n✅ Central Kaplan-Meier test completed successfully.")
