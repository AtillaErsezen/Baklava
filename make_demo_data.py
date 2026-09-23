"""
Synthetic demo datasets for ML Factory, written to data/. Deterministic (seed 42).

  uv run make_demo_data.py

data/churn.csv   3000 rows, binary classification, target `Churn` (Yes/No, ~20% Yes).
    Churn depends on monthly_charges, tenure_months, support_tickets and contract.
    Built-in traps for the agent to catch:
      - customer_id    unique integer per row -> profiled as `id_like`
      - refund_issued  equals the churn label in 99% of rows (information from after the
                       customer left) -> profiled as `possible_leakage`. Left in, it
                       inflates CV scores to near-perfect.
      - last_login_days  ~10% missing, exercises imputation

data/houses.csv  2000 rows, regression, target `SalePrice`.
    Price mainly depends on sqft times a neighborhood multiplier, minus age, plus bedrooms,
    with Gaussian noise. lot_size (~15% missing) and garage are pure noise features.
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
os.makedirs("data", exist_ok=True)

# churn: binary classification; `refund_issued` is the planted leak (set after the customer churned)
n = 3000
tenure = rng.integers(1, 72, n)
monthly = rng.normal(70, 25, n).clip(15, 150).round(2)
contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2])
support = rng.poisson(1.5, n)
logit = -1 + 0.03 * (monthly - 70) - 0.05 * tenure + 0.4 * support + np.where(contract == "month-to-month", 1.2, -0.8)
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
pd.DataFrame({
    "customer_id": np.arange(100000, 100000 + n),
    "tenure_months": tenure,
    "monthly_charges": monthly,
    "contract": contract,
    "payment_method": rng.choice(["card", "bank", "check", "paypal"], n),
    "support_tickets": support,
    "has_streaming": rng.random(n) < 0.4,
    "last_login_days": np.where(rng.random(n) < 0.1, np.nan, rng.exponential(10, n).round()),
    "refund_issued": np.where(rng.random(n) < 0.99, churn, 1 - churn),
    "Churn": np.where(churn == 1, "Yes", "No"),
}).to_csv("data/churn.csv", index=False)

# houses: regression
n = 2000
sqft = rng.normal(1800, 600, n).clip(400, 6000).round()
beds = (sqft / 600 + rng.normal(0, 0.8, n)).clip(1, 7).round().astype(int)
age = rng.integers(0, 100, n)
hood = rng.choice(["north", "south", "east", "west", "center"], n)
hood_mult = pd.Series(hood).map({"north": 1.1, "south": 0.85, "east": 0.95, "west": 1.0, "center": 1.35}).to_numpy()
price = (sqft * 150 * hood_mult - age * 800 + beds * 5000 + rng.normal(0, 25000, n)).clip(30000).round(-2)
pd.DataFrame({
    "sqft": sqft, "bedrooms": beds, "age_years": age, "neighborhood": hood,
    "garage": rng.random(n) < 0.6,
    "lot_size": np.where(rng.random(n) < 0.15, np.nan, rng.normal(6000, 2000, n).clip(1000).round()),
    "SalePrice": price,
}).to_csv("data/houses.csv", index=False)

print("wrote data/churn.csv, data/houses.csv")
