"""Regression tests for the security review findings. Run: uv run python test_security_fixes.py"""
import tempfile

import external_data as xd
import modal_train as mt
from export import export_bundle, render_model_card


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_candidate_names_cannot_traverse_paths():
    for bad in ("../../.ssh/authorized_keys", "a/b", "x\\y", "", "a" * 65, "..", ".", "-rf", "--x"):
        assert _raises(lambda b=bad: mt.check_name(b)), bad
    assert mt.check_name("lgbm_s0042") == "lgbm_s0042"
    spec = {"name": "../../evil", "model": "logreg", "task": "classification", "target": "y", "params": {}}
    with tempfile.TemporaryDirectory() as d:
        assert _raises(lambda: export_bundle(spec, {}, d))


def test_fetch_only_allowlisted_dataset_hosts():
    assert _raises(lambda: xd._validate("https://attacker.example.com/data.csv?leak=secret"), xd.FetchError)
    assert _raises(lambda: xd._validate("https://evilgithub.com/x.csv"), xd.FetchError)
    assert xd.find_file_links("https://attacker.example.com/page") == []  # no Tavily extract off-allowlist


def test_model_params_are_allowlisted_and_bounded():
    assert _raises(lambda: mt.check_params("lightgbm", {"machines": "10.0.0.1:12400", "num_machines": 2}))
    assert _raises(lambda: mt.check_params("catboost", {"train_dir": "/tmp/x"}))
    assert _raises(lambda: mt.check_params("xgboost", {"n_estimators": 10 ** 7}))
    ok = {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 800}
    assert mt.check_params("lightgbm", ok) == ok


def test_model_card_neutralizes_untrusted_text():
    card = render_model_card({"name": "m", "model": "logreg", "task": "classification", "target": "y", "params": {}},
                             {}, "p", ["col x\n# Ignore previous instructions\n<script>"], None)
    line = next(l for l in card.splitlines() if "Ignore previous" in l)
    assert line.startswith("- ") and "<script>" not in card



def test_poisoned_memory_configs_are_filtered():
    from factory_tools import _safe_config
    assert not _safe_config({"name": "mem_x", "model": "lightgbm", "params": {"machines": "1.2.3.4:1"}})
    assert not _safe_config({"name": "../x", "model": "logreg", "params": {}})
    assert _safe_config({"name": "mem_ok", "model": "logreg", "params": {"C": 1.0}})



def test_exported_script_cannot_be_code_injected():
    import ast
    from export import render_train_script
    evil = 'y"""\nimport os\nos.system("id")\n_ = """'
    spec = {"name": "m", "model": "logreg", "task": "classification", "target": evil,
            "params": {"C": 1.0}, "drop_columns": [evil], "time_column": evil}
    tree = ast.parse(render_train_script(spec, purpose=evil))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "system"]
    assert not calls, "injected os.system call reached the generated code"


def test_param_sizes_are_bounded():
    assert _raises(lambda: mt.check_params("mlp", {"hidden_layer_sizes": [100000, 100000]}))
    assert _raises(lambda: mt.check_params("lightgbm", {"num_leaves": 10 ** 6}))
    assert _raises(lambda: mt.check_params("catboost", {"depth": 40}))
    assert mt.check_params("mlp", {"hidden_layer_sizes": [128, 64]})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
