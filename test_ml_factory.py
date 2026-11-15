"""Offline checks, no network: `uv run test_ml_factory.py`. Needs data/ (`uv run make_demo_data.py`).
The last check runs the full agent loop with --no-llm."""
import glob
import os

import numpy as np
import pandas as pd

import agent
import stats
import train

churn = train.load_local("data/churn.csv")


def test_split():
    dev, hidden = agent.split_holdout(churn, "Churn", "classification")
    assert len(hidden) == 600 and len(dev) == 2400
    assert not set(dev.customer_id) & set(hidden.customer_id)
    assert abs((dev.Churn == "Yes").mean() - (hidden.Churn == "Yes").mean()) < 0.01

    t = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=100), "x": range(100), "y": [0, 1] * 50})
    col = agent.sorted_time_column(t, "y")
    dev, hidden = agent.split_holdout(t, "y", "classification", col)
    assert col == "when" and dev.when.max() < hidden.when.min() and len(hidden) == 20


def test_profile_flags():
    dev, _ = agent.split_holdout(churn, "Churn", "classification")
    flags = {c["name"]: c.get("flags", []) for c in agent.profile_data(dev, "Churn")["columns"]}
    assert "possible_leakage" in flags["refund_issued"] and "id_like" in flags["customer_id"]
    assert not flags["contract"] and not flags["monthly_charges"]

    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b"], 1000)
    status = np.where(rng.random(1000) < 0.995, np.where(y == "a", "closed", "open"), "open")
    df = pd.DataFrame({"status": status, "noise": rng.choice(["p", "q", "r"], 1000), "y": y})
    flags = {c["name"]: c.get("flags", []) for c in agent.profile_data(df, "y")["columns"]}
    assert "possible_leakage" in flags["status"] and not flags["noise"]

    assert agent.cross_split_duplicates(churn, churn.head(5), "Churn") == 5


def test_stats():
    t, p = stats.nb_corrected_t([1, 2, 3, 4], [0, 0, 0, 0], 1.0)  # mean 2.5, var 5/3 → t = √3
    assert abs(t - 3 ** 0.5) < 1e-9 and 0.08 < p < 0.10
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    rng = np.random.default_rng(1)
    y = rng.normal(0, 5, 2000)
    out = stats.paired_bootstrap(y, {"a": y + rng.normal(0, 1, 2000), "b": y + rng.normal(0, 2, 2000)}, "rmse", "a",
                                 n_boot=500)
    lo, hi = out["b"]["diff_ci"]
    assert 0 < lo < 1 < hi, out["b"]  # true rmse diff ≈ 1, clearly above 0
    assert out["a"]["diff_ci"] == [0.0, 0.0]


def test_train_core():
    dev, hidden = agent.split_holdout(churn, "Churn", "classification")
    spec = {"name": "lr", "model": "logreg", "task": "classification", "target": "Churn", "cv": "kfold",
            "cv_folds": 2, "cv_repeats": 5, "drop_columns": ["customer_id", "refund_issued"]}
    r = train._train(dev, spec)
    assert r["ok"], r
    assert len(r["metrics"]["roc_auc"]["folds"]) == 10 and 0.6 < r["metrics"]["roc_auc"]["mean"] < 0.95
    h = train._predict_holdout(dev, hidden, spec)
    assert h["ok"] and len(h["y"]) == len(h["pred"]) == 600, h
    assert stats.score("roc_auc", h["y"], h["pred"]) > 0.6
    assert not train._train(dev, {**spec, "model": "nope"})["ok"]


def test_no_llm_loop():
    run = agent.FactoryRun("data/churn.csv", "Churn", no_llm=True)
    try:
        run.run()
        kinds = [e["kind"] for e in run.events]
        assert {"candidate", "leaderboard", "confirm", "hidden_test", "final_model", "report"} <= set(kinds), kinds
        assert set(run.specs["logreg_default"]["drop_columns"]) == {"customer_id", "refund_issued"}
        assert all(r["ok"] for r in run.results), [r.get("error") for r in run.results]
        tie = [f["name"] for f in run.finalists if f["tie_with_best"]]
        assert run.report["recommended"] in tie and "hidden" in run.finalists[0]
        assert "| model |" in run.report["markdown"]
        # a later round that leaves out drop_columns keeps the earlier drops (the leaky column stays out)
        out = run.tool_run_experiments({"rationale": "t", "task": "classification", "primary_metric": "roc_auc",
                                        "candidates": [{"name": "lr_again", "model": "logreg"}]})
        assert set(run.specs["lr_again"]["drop_columns"]) == {"customer_id", "refund_issued"}
        assert out["leaderboard"][0]["roc_auc"] < 0.95, out
        pred = train.predict(run.final["model_path"], churn.head(50).drop(columns="Churn"))
        assert len(pred) == 50 and set(pred) <= {"Yes", "No"}
    finally:
        for f in glob.glob(f"runs/{run.run_id}_*"):
            os.remove(f)


def test_script():
    import ast
    import subprocess
    import sys
    import tempfile

    spec = {"name": "lr", "model": "logreg", "task": "classification", "target": "Churn", "params": {"C": 0.5},
            "preprocessing": {"categorical": "ordinal"}, "drop_columns": ["customer_id", "refund_issued"]}
    with tempfile.TemporaryDirectory() as tmp:
        out = train.fit_final(churn, spec, f"{tmp}/m.joblib")
        with open(f"{tmp}/model.py", "w") as f:
            f.write(out["script"])
        data = os.path.abspath("data/churn.csv")
        subprocess.run([sys.executable, "model.py", data, data], cwd=tmp, check=True, capture_output=True)
        # the copied script, run on its own, predicts exactly what the saved model does
        assert pd.read_csv(f"{tmp}/predictions.csv")["prediction"].tolist() == train.predict(f"{tmp}/m.joblib", churn)

    # LLM-chosen names and params cannot inject code into the script
    evil = {**spec, "model": "lightgbm", "name": "x\nimport os", "params": {"a); import os  #": 1}}
    tree = ast.parse(train.script(evil, churn))
    assert "os" not in {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def test_ui_server():
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    import ui

    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    def call(path, data=None, headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers or {})) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    ui.UPLOADS = "runs/_test_uploads"
    try:
        assert call("/events")[1] == {"events": [], "running": False, "error": None}
        assert call("/config", headers={"Host": "evil.example"})[0] == 403
        assert call("/stop", b"", {"Origin": "http://evil.example"})[0] == 403
        assert call("/upload", b"x", {"X-Filename": "notes.txt"})[0] == 400
        code, up = call("/upload", b"a,b,y\n1,2,0\n3,4,1\n", {"X-Filename": "../../tiny.csv"})
        assert code == 200 and up == {"file": "tiny.csv", "rows": 2, "columns": ["a", "b", "y"]}, up
        assert os.path.isfile("runs/_test_uploads/tiny.csv")  # path stripped, stays inside UPLOADS
        assert call("/run", json.dumps({"file": "missing.csv", "target": "y"}).encode())[0] == 400
    finally:
        srv.shutdown()
        for f in glob.glob("runs/_test_uploads/*"):
            os.remove(f)


def test_llm_scores():
    import llm_train as lt

    assert lt.token_f1("The cat sat.", "the cat sat") == 1.0 and lt.token_f1("dog", "cat") == 0.0
    assert abs(lt.token_f1("the cat", "the cat sat down") - 2 / 3) < 1e-9  # precision 1, recall 0.5
    assert lt.wer(["the cat sat"], ["the cat sat"]) == 0.0
    assert abs(lt.wer(["the bat sat down"], ["the cat sat"]) - 2 / 3) < 1e-9  # 1 substitution + 1 insertion / 3
    qs = {"cat": {"type": "choice", "options": ["a", "b"]}, "ok": {"type": "noul"},
          "pri": {"type": "score", "min": 0, "max": 10}}
    score, per_q = lt.decision_agreement([{"cat": "a", "ok": True, "pri": 5.0}, {"cat": "b", "ok": True, "pri": 0.0}],
                                         [{"cat": "a", "ok": "yes", "pri": 5}, {"cat": "a", "ok": "no", "pri": 10}], qs)
    assert per_q == {"cat": 0.5, "ok": 0.5, "pri": 0.5} and score == 0.5
    assert lt.pick_winner([{"ok": True, "score": 0.2, "name": "a"}, {"ok": False, "name": "b"},
                           {"ok": True, "score": 0.7, "name": "c"}])["name"] == "c"
    assert lt.pick_winner([{"ok": False}]) is None
    assert set(lt.PORTFOLIO) == set(lt.DEFAULTS) == set(lt.BUILDERS) == set(lt.KINDS)


def test_llm_setup():
    import llm_factory as lf

    guess = lambda d, media=None: lf.infer_setup(pd.DataFrame(d), media)
    t = guess({"instruction": ["hi", "bye"] * 10, "output": ["hello", "ciao"] * 10})
    assert t["kind"] == "text" and t["columns"] == {"prompt": "instruction", "response": "output"}
    d = guess({"text": [f"ticket {i} about a long enough problem" for i in range(40)],
               "category": ["billing", "tech"] * 20, "urgent": ["yes", "no"] * 20, "priority": [i * 2.5 for i in range(40)]})
    assert d["kind"] == "decision" and d["columns"]["inputs"] == ["text"]
    assert d["questions"] == {"category": {"type": "choice", "options": ["billing", "tech"]}, "urgent": {"type": "noul"},
                              "priority": {"type": "score", "min": 0.0, "max": 97.5}}
    assert guess({"image": ["a.png", "b.jpg"], "question": ["q"] * 2, "answer": ["a"] * 2})["kind"] == "vision"
    assert guess({"audio": ["a.wav", "b.wav"], "text": ["x", "y"]})["kind"] is None  # stt or tts: the goal decides
    one_class = guess({"text": ["a long review text"] * 4, "label": ["neg"] * 4}, {"label": ["neg", "pos"]})
    assert one_class["questions"] == {"label": {"type": "choice", "options": ["neg", "pos"]}}


def test_llm_notebook_and_run():
    import ast
    import base64
    import json
    import tempfile

    import llm_factory as lf
    import llm_train as lt

    with tempfile.TemporaryDirectory() as tmp:
        data = f"{tmp}/chats.jsonl"
        with open(data, "w") as f:
            f.writelines(json.dumps({"prompt": f"q{i}", "response": f"a{i}"}) + "\n" for i in range(30))

        # --kaggle: no training, a notebook that carries the trainer, the data and the plan
        run = lf.LLMFactoryRun(data, "answer questions", no_llm=True, kaggle=f"{tmp}/run.ipynb")
        run.run()
        with open(f"{tmp}/run.ipynb") as f:
            nb = json.load(f)
        code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        trainer = next(c for c in code if c.startswith("%%writefile llm_train.py\n"))
        with open(lt.__file__) as f:
            assert trainer.split("\n", 1)[1] == f.read()
        scope = {}
        exec(next(c for c in code if c.startswith("PLAN = ")), scope)  # the plan cell is plain Python literals
        assert scope["PLAN"]["kind"] == "text" and scope["PLAN"]["data"] == "/kaggle/working/chats.jsonl"
        assert scope["PLAN"]["candidates"] == lt.PORTFOLIO["text"]
        payload = ast.parse(next(c for c in code if c.startswith("import base64"))).body[1].body[0].value.args[0].args[0]
        with open(data, "rb") as f:
            assert base64.b64decode(payload.value) == f.read()
        assert "run_plan(PLAN" in code[-1] and nb["metadata"]["kaggle"]["isGpuEnabled"]
        for f in glob.glob(f"runs/{run.run_id}_*"):
            os.remove(f)

        # local run with the trainer replaced: the best test score wins, whatever order results arrive in
        scores = {"smollm2-360m": 0.31, "qwen2.5-0.5b": 0.58, "qwen2.5-1.5b": None}
        real = lt.train_all, lt.export_winner

        def fake_train_all(plan, out_dir, parallel=None, on_result=None):
            assert plan["columns"]["prompt"] == "prompt" and plan["data"] == data
            results = [{"name": c["name"], "ok": True, "base_model": c["base_model"], "score": scores[c["name"]],
                        "metrics": {"token_f1": scores[c["name"]]}, "train_seconds": 1.0, "model_dir": tmp}
                       if scores[c["name"]] is not None else
                       {"name": c["name"], "ok": False, "base_model": c["base_model"], "error": "CUDA out of memory"}
                       for c in plan["candidates"]]
            for r in results:
                on_result(r)
            return results

        lt.train_all = fake_train_all
        lt.export_winner = lambda winner, plan, out_dir, losers=(): {"model_dir": tmp, "zip": f"{tmp}/m.zip",
                                                                     "ollama": True, "note": None}
        run = lf.LLMFactoryRun(data, "answer questions", no_llm=True)
        try:
            run.run()
            assert run.final["name"] == "qwen2.5-0.5b" and run.report["recommended"] == "qwen2.5-0.5b"
            assert "| qwen2.5-0.5b ★ |" in run.report["markdown"] and "| failed |" in run.report["markdown"]
            assert [e["kind"] for e in run.events].count("candidate") == 3
            assert "error" in run.tool_train_candidates({"rationale": "t", "kind": "decision", "candidates": []})  # kind is fixed
        finally:
            lt.train_all, lt.export_winner = real
            for f in glob.glob(f"runs/{run.run_id}_*"):
                os.remove(f)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
