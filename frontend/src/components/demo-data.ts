// Dataset dimensions match the repository fixtures. Scores and conclusions are illustrative.
export const examples = {
  churn: {
    label: "Customer churn",
    file: "churn.csv",
    task: "Classification",
    rows: "3,000",
    features: 9,
    target: "Churn",
    metric: "ROC AUC",
    winner: "LightGBM",
    question: "Which customers are likely to leave?",
    insight:
      "In this example, LightGBM offers the strongest validation score. Tenure and support tickets help explain the predictions.",
    models: [
      { name: "LightGBM", score: "0.924", width: 92.4 },
      { name: "XGBoost", score: "0.917", width: 91.7 },
      { name: "Random Forest", score: "0.896", width: 89.6 },
      { name: "Logistic Regression", score: "0.851", width: 85.1 },
    ],
  },
  houses: {
    label: "House prices",
    file: "houses.csv",
    task: "Regression",
    rows: "2,000",
    features: 6,
    target: "SalePrice",
    metric: "R²",
    winner: "XGBoost",
    question: "What makes a house worth more?",
    insight:
      "In this example, XGBoost explains the most variation in house prices. Floor area and neighborhood are the strongest signals.",
    models: [
      { name: "XGBoost", score: "0.938", width: 93.8 },
      { name: "LightGBM", score: "0.929", width: 92.9 },
      { name: "Random Forest", score: "0.901", width: 90.1 },
      { name: "Ridge", score: "0.864", width: 86.4 },
    ],
  },
};
export type ExampleKey = keyof typeof examples;
