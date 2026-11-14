"""Predict with a saved model: uv run predict.py runs/<run_id>_model.joblib new.csv [out.csv]"""
import sys

import train

model, data, *out = sys.argv[1:]
df = train.load_local(data)
df["prediction"] = train.predict(model, df)
df.to_csv(out[0] if out else "predictions.csv", index=False)
print("wrote", out[0] if out else "predictions.csv")
