"""Windows-specific failure modes, reproduced on any OS. Run: uv run python -m pytest -q test_windows.py"""
import os
import subprocess
import sys


def test_non_ascii_print_survives_a_cp1252_console():
    """Windows redirects stdout as cp1252; an LLM thought with '≥' or '±' must not crash a run."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    code = "import agent, json; print(json.dumps({'text': 'AUC ≥ 0.8 ± 0.01 → ok'}, ensure_ascii=False))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-500:]


def test_generated_files_are_utf8_with_no_platform_newline_surprises(tmp_path):
    from export import export_bundle
    spec = {"name": "m1", "model": "logreg", "task": "classification", "target": "y", "params": {}}
    paths = export_bundle(spec, {"cv_mean": 0.8}, str(tmp_path), purpose="Qualité ≥ 0.8")
    for p in paths.values():
        with open(p, encoding="utf-8") as f:  # must decode as UTF-8 regardless of the OS locale
            assert f.read()
