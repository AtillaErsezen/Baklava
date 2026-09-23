"""Offline check of the agent loop: scripted LLM, Modal stubbed. Run: uv run python test_agent_loop.py"""
import agent


class FakeFn:
    def map(self, specs):
        return [{"name": s["name"], "ok": True, "fit_seconds": 0.1,
                 "metrics": {"roc_auc": {"mean": 0.8, "std": 0.01, "train_mean": 0.85}}} for s in specs]

    def remote(self, spec, run_id):
        return {"model_path": "/models/x.joblib", "n_rows": 10, "top_features": []}


def test_scripted_run_reaches_report():
    agent.modal.Function.from_name = lambda app, name: FakeFn()
    agent.upload_dataset = lambda df: "/datasets/fake.parquet"
    run = agent.FactoryRun("data/churn.csv", "Churn", provider="scripted")
    run.run()
    kinds = [e["kind"] for e in run.events]
    assert kinds[-2:] == ["report", "usage"] and "leaderboard" in kinds and run.final
    dropped = run.specs["logreg"]["drop_columns"]
    assert "refund_issued" in dropped and "customer_id" in dropped


if __name__ == "__main__":
    test_scripted_run_reaches_report()
    print("ok test_scripted_run_reaches_report")
