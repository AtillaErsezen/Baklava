# ML Factory — local single agent

Give it a table and a target column, tell it what the model is for, and it finds the best model on your own machine.
Nothing leaves your computer unless you point it at a remote LLM.

What a run does:

1. Holds back 20% of the rows as a hidden test set.
2. Profiles the data and drops leaky, id-like and constant columns.
3. Cross-validates a fixed portfolio of 15 configs: logistic/ridge regression, random forest, LightGBM, XGBoost, MLP.
4. The agent tunes the best families for up to two more rounds (hyperparameters, preprocessing, class weights).
5. Re-tests the top 3 with repeated CV and a significance test, then scores them once on the hidden set.
6. Fits the recommended model on all rows and saves it with a report.

## Run

```bash
uv sync
uv run make_demo_data.py   # optional: writes data/churn.csv and data/houses.csv
```

In the browser:

```bash
uv run ui.py
```

Then open http://127.0.0.1:8000, add your table, pick the column to predict and start the run. The page shows
every model as it finishes, the final three, and a download button for the winner.

From the terminal, with a local LLM ([Ollama](https://ollama.com), or any OpenAI-compatible server):

```bash
ollama pull qwen3:4b-instruct
OLLAMA_CONTEXT_LENGTH=16384 ollama serve   # skip if Ollama already runs as a service with a large enough context
uv run agent.py data/churn.csv --target Churn --goal "catch churners early; a missed churner costs more than a false alarm"
```

Choosing a model:

- It must support tool calling and should not be a "thinking" model. Thinking variants (plain `qwen3`, for example)
  spend hundreds of tokens before every step.
- The agent's prompt, the data profile and the leaderboards need about 8k tokens of context. Ollama's default can be
  smaller, and then the model silently forgets its instructions.
- Speed depends on your hardware. Without a GPU a 4B model manages a few tokens per second, so a run takes tens of
  minutes; on a thin laptop it can be slower still. `--no-llm` finishes in about a minute.

Without an LLM (fixed portfolio only: no tuning rounds, no written narrative):

```bash
uv run agent.py data/houses.csv --target SalePrice --no-llm
```

| Option | Meaning |
|---|---|
| `data` | CSV or Parquet file |
| `--target` | column to predict |
| `--task` | `classification` or `regression`; guessed if omitted |
| `--goal` | free text: what the model is for, constraints, which errors matter |
| `--no-llm` | skip the LLM |

| Env var | Default |
|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` |
| `AGENT_MODEL` | `qwen3:4b-instruct` |
| `LLM_API_KEY` | not needed for local servers |

## Output

Everything lands in `runs/`:

- `<run_id>_model.joblib`: the winning model
- `<run_id>_report.md`: the report with the finalist table
- `<run_id>_results.json`, `<run_id>_events.jsonl`: every score and step

Use the model on new rows:

```bash
uv run predict.py runs/<run_id>_model.joblib new.csv predictions.csv
```

## Tests

```bash
uv run test_ml_factory.py
```
