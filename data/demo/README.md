# Demo datasets (real, public)

| File | Target | Paste this goal |
|---|---|---|
| `telco_churn.csv` (IBM Telco, 7043 rows) | `Churn` | Which customers will churn soon? Explainable is a plus. |
| `walmart_sales.csv` (45 stores, weekly) | `Weekly_Sales` | Forecast next week's sales per store. |
| `../churn.csv` (synthetic, planted leak) | `Churn` | Flag likely churners early. |

Telco: the agent should flag `customerID` as an id and read the blank `TotalCharges` cells as missing.
