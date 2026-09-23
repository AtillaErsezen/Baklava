import { parseRun } from "./data";
import type { RunEvent } from "./data";
function sample(regression = false) {
  const id = regression ? "sample-houses" : "sample-churn";
  const metric = regression ? "rmse" : "roc_auc";
  const names = regression
    ? ["xgboost", "lightgbm", "random_forest", "ridge"]
    : ["lightgbm", "xgboost", "random_forest", "logreg"];
  const scores = regression
    ? [28450, 29120, 33800, 41600]
    : [0.924, 0.917, 0.896, 0.851];
  const fits = [2.4, 3.8, 8.2, 0.6];
  const features = regression
    ? ["Floor area", "Neighborhood", "Quality", "Year built", "Lot area"]
    : [
        "Tenure",
        "Support tickets",
        "Monthly spend",
        "Contract type",
        "Payment method",
      ];
  const summary = regression
    ? "XGBoost gives the lowest prediction error in this sample. Floor area and neighborhood carry the strongest signal."
    : "LightGBM finds the strongest signal in this sample. Customer tenure and support tickets contribute most to its predictions.";
  const event = (
    kind: string,
    payload: Record<string, unknown>,
    offset: number,
  ): RunEvent => ({ run_id: id, ts: 1790150400 + offset, kind, payload });
  const events = [
    event(
      "run_start",
      {
        dataset: regression ? "houses.csv" : "churn.csv",
        target: regression ? "SalePrice" : "Churn",
        rows: regression ? 2000 : 3000,
        features: regression ? 6 : 9,
      },
      0,
    ),
    event(
      "thought",
      {
        text: "First, I’ll profile the dataset and check for missing values, imbalance, and possible leakage.",
      },
      1,
    ),
    event(
      "diagnostics",
      {
        findings: [
          {
            check: "leakage",
            severity: 2,
            finding:
              "Review features collected after the outcome. Remove potential leakage before a real training run.",
          },
          {
            check: "holdout",
            severity: 1,
            finding:
              "Reserve an unseen holdout to check how well the selected model generalizes.",
          },
        ],
      },
      4,
    ),
    event(
      "search_plan",
      {
        space: 3024,
        raced: 243,
        rationale:
          "Compare established model families and progressively focus on the most promising candidates.",
      },
      6,
    ),
    event(
      "thought",
      {
        text: "The leading tree models show a useful signal. I’ll compare the finalists on paired validation folds.",
      },
      96,
    ),
    event(
      "confirm",
      {
        primary_metric: metric,
        recommendation: { pick: names[0] },
        table: names.map((name, i) => ({
          name,
          family: name,
          cv_mean: scores[i],
          ci: regression
            ? [scores[i] - 1100, scores[i] + 1100]
            : [scores[i] - 0.008, scores[i] + 0.008],
          hidden: regression ? scores[i] + 820 : scores[i] - 0.006,
          fit_s: fits[i],
          predict_ms: 0.08 + i * 0.04,
          params:
            i === 3
              ? regression
                ? { alpha: 1 }
                : { C: 1, max_iter: 1000 }
              : {
                  n_estimators: 200,
                  learning_rate: 0.05,
                  max_depth: i === 0 ? 6 : 4,
                },
        })),
      },
      138,
    ),
    event(
      "final_model",
      {
        name: names[0],
        spec: {
          task: regression ? "regression" : "classification",
          model: names[0],
        },
        top_features: features.map((feature, i) => ({
          feature,
          importance: [0.36, 0.27, 0.18, 0.12, 0.07][i],
        })),
      },
      142,
    ),
    event(
      "report",
      {
        spoken_summary: summary,
        markdown: `## The recommendation\n\n${summary}\n\n## What the comparison tells us\n\n${regression ? "The selected model has a sample cross-validation RMSE of 28,450. Lower error is better. A held-out score of 29,270 suggests a similar error on unseen examples." : "The selected model has a sample cross-validation ROC AUC of 0.924 and a held-out score of 0.918. ROC AUC measures how well a model ranks positive examples above negative ones; it is not percentage accuracy."}\n\n## Before putting it to work\n\n- Check that every input feature is available at prediction time.\n- Validate on a representative, unseen dataset.\n- Review performance across relevant groups.\n\nThese results are illustrative. No live training was performed.`,
      },
      143,
    ),
    event("usage", { usd: 0.04 }, 144),
  ];
  const run = parseRun(JSON.stringify(events));
  return {
    ...run,
    sample: true,
    label: regression ? "House prices" : "Customer churn",
  };
}
export const samples = [sample(), sample(true)];
