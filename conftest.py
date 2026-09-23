"""pytest setup shared by every test file: run from the repo root and make sure runs/ exists
(it is gitignored, so a fresh clone on any OS does not have it)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
