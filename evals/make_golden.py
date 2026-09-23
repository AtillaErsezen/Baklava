"""Golden eval datasets with known answers -> evals/data/. Run: uv run evals/make_golden.py

Each dataset plants one property the agent must detect or handle. EXPECT (below) is what
evals/run_golden.py asserts after a run."""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "data")
rng = np.random.default_rng(7)

EXPECT = {
    # name: target, task, the diagnostic that must fire (or None), columns that must be dropped, cv scheme
    "leaky_churn": {"target": "Churn", "task": "classification", "finding": "leakage",
                    "must_drop": ["refund_issued"], "cv": "kfold"},
    "clean_houses": {"target": "SalePrice", "task": "regression", "finding": None, "must_drop": [], "cv": "kfold"},
    "drifting_sales": {"target": "units", "task": "regression", "finding": "adversarial_drift", "must_drop": [],
                       "cv": ("walk_forward", "purged")},
    "high_card_claims": {"target": "fraud", "task": "classification", "finding": "high_cardinality",
                         "must_drop": [], "cv": "kfold"},
    "tiny_diabetes": {"target": "outcome", "task": "classification", "finding": None, "must_drop": [], "cv": "kfold"},
}


def leaky_churn(n=3000):
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150).round(2)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2])
    logit = -1 + 0.03 * (monthly - 70) - 0.05 * tenure + np.where(contract == "month-to-month", 1.2, -0.8)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"tenure_months": tenure, "monthly_charges": monthly, "contract": contract,
                         "refund_issued": np.where(rng.random(n) < 0.99, churn, 1 - churn),
                         "Churn": np.where(churn == 1, "Yes", "No")})


def clean_houses(n=2000):
    sqft = rng.normal(1800, 600, n).clip(400, 6000).round()
    age = rng.integers(0, 100, n)
    hood = rng.choice(["north", "south", "east", "west"], n)
    mult = pd.Series(hood).map({"north": 1.1, "south": 0.85, "east": 0.95, "west": 1.0}).to_numpy()
    price = (sqft * 150 * mult - age * 800 + rng.normal(0, 25000, n)).clip(30000).round(-2)
    return pd.DataFrame({"sqft": sqft, "age_years": age, "neighborhood": hood, "SalePrice": price})


def drifting_sales(n=1500):
    day = pd.date_range("2022-01-01", periods=n, freq="D")
    trend = np.linspace(0, 3, n)  # the relationship and the inputs drift over time
    price = rng.normal(10, 2, n) + trend
    promo = rng.integers(0, 2, n)
    units = (50 - 2.5 * price + 8 * promo + 6 * np.sin(np.arange(n) * 2 * np.pi / 7) + 4 * trend
             + rng.normal(0, 3, n)).round(1)
    return pd.DataFrame({"date": day, "price": price.round(2), "promo": promo, "units": units})


def high_card_claims(n=4000):
    provider = rng.integers(0, 900, n).astype(str)  # ~900 levels: must be handled, not one-hot to 900 columns
    bad = set(rng.choice(900, 40, replace=False).astype(str))
    amount = rng.lognormal(7, 1, n).round(2)
    logit = -3.2 + 2.5 * pd.Series(provider).isin(bad).to_numpy() + 0.4 * (np.log(amount) - 7)
    fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"provider_id": "P" + pd.Series(provider), "amount": amount,
                         "channel": rng.choice(["web", "phone", "agent"], n), "fraud": fraud})


def tiny_diabetes(n=400):
    glucose = rng.normal(120, 30, n).round()
    bmi = rng.normal(31, 6, n).round(1)
    age = rng.integers(21, 80, n)
    logit = -8 + 0.04 * glucose + 0.08 * bmi + 0.02 * age
    return pd.DataFrame({"glucose": glucose, "bmi": bmi, "age": age,
                         "outcome": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in EXPECT:
        df = globals()[name]()
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"wrote {name}.csv {df.shape}")
