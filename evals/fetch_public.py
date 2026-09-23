"""Download the public eval datasets (evals/public_catalog.py) into evals/public/<name>.csv.

Run: uv run evals/fetch_public.py
Every download goes through external_data.safe_fetch, so the https-only, allowlisted-host, public-IP and
size-cap guards apply. Files already on disk are skipped. Data is saved as fetched, traps included.

Traps worth knowing (checked on the fetched files and with the diagnostics battery on the dev split):
- telco_churn: TotalCharges is a blank string for the 11 tenure-0 customers; modal_train.load_local casts it to
  numbers with 11 NaN. customerID is a unique id (id_like), left out of n_over_p's feature count.
- titanic: PassengerId and Name are unique per row. Cabin is 77% missing and its missingness predicts survival.
  n_over_p leaves out the ids and caps Ticket and Cabin at the one-hot width the pipeline uses.
- california_housing: rows are ordered by location, so adversarial_drift (severity 3) and autocorr_target fire
  with no time axis at all; kfold is still right. The target is capped at 500001 (965 censored rows).
- bank_marketing: duration (call length) is only known after the call and nearly decides y; UCI says to drop it
  for a real model. No check flags it, so dropping it is pure judgment. id is a row id.
- adult_income: missing values are the string "?" (workclass, occupation, native.country); missing_pattern
  stays silent, missing_placeholders names them. Rows are ordered (target lag-1 autocorrelation 0.27).
- credit_default: ID is a row id. EDUCATION has undocumented codes 0, 5, 6; the PAY columns skip PAY_1.
- bike_sharing_hourly: casual + registered = cnt on every row, a perfect leak. leakage names the pair
  (registered + casual, severity 3) from an exact least-squares fit. instant is a row index; dteday is sorted,
  so the split is by time.
- walmart_sales: Date is dd-mm-yyyy; modal_train.load_local detects day-first from values like 19-02-2010.
  Weekly rows all fall on a Friday. Rows are grouped by Store, which drives adversarial_drift.
- pima_diabetes: zeros mean missing in Insulin (374), SkinThickness (227), BloodPressure (35), BMI (11) and
  Glucose (5). Only mad_outliers notices anything.
- card_fraud_sample: a biased sample. All 897 legit rows come from the first 718 seconds, the 51 frauds from
  later, so Time alone scores AUC 0.97, a sampling artifact. The leakage check names V12 (0.985) instead, which
  is real signal in this fraud-enriched sample: drop Time, not V12.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from collections.abc import Callable

import pandas as pd

import external_data as xd
from evals.public_catalog import PUBLIC, PUBLIC_DIR

TIMEOUT_S = 120


def fetch(name: str, entry: dict, out_dir: str = PUBLIC_DIR,
          fetcher: Callable[..., pd.DataFrame] = xd.safe_fetch) -> tuple[str, str]:
    """Save one catalog entry as out_dir/<name>.csv unless it exists; return (path, "cached" or "fetched")."""
    path = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "cached"
    df = fetcher(entry["url"], timeout=TIMEOUT_S)
    if entry["target"] not in df.columns:
        raise ValueError(f"{name}: target {entry['target']!r} not in fetched columns {list(df.columns)[:20]}")
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # no half-written file survives a crash
    return path, "fetched"


def main() -> int:
    """Fetch every entry, print a markdown summary, return 1 if any failed."""
    lines, failed = ["| dataset | rows | cols | status | source |", "|---|---|---|---|---|"], 0
    for name, entry in PUBLIC.items():
        try:
            path, status = fetch(name, entry)
            rows, cols = pd.read_csv(path, low_memory=False).shape
            lines.append(f"| {name} | {rows} | {cols} | {status} | {entry['source']} |")
        except (xd.FetchError, ValueError, OSError) as e:
            failed += 1
            lines.append(f"| {name} | - | - | FAILED: {str(e)[:120]} | {entry['source']} |")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
