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

## LLM Factory: fine-tune open models

`llm_factory.py` does for language, vision and speech models what the tabular agent does for tables: you give it
data and a goal, a crafter agent picks open models from Hugging Face, several candidates are fine-tuned in parallel,
each is scored on the same held-out test rows, and the one with the best test score is exported.

> Status: the training code in `llm_train.py` has not been executed yet. Data profiling, Hub search, the run flow
> and the Kaggle export are tested; the first real training run is the test of the rest.

| Kind | What it trains | Data columns | Test score |
|---|---|---|---|
| `text` | chat / instruction LLM, LoRA | `prompt` + `response`, or `messages` | word-overlap F1 of generated answers |
| `decision` | "System One" model: typed decisions (choice, yes/no, score) from one encoder pass, no text generation | input text + one column per decision | agreement with the labels |
| `vision` | vision-language model, LoRA | `image` + `prompt` + `response` | word-overlap F1 of generated answers |
| `stt` | speech-to-text, Whisper family | `audio` + `text` | 1 − word error rate |
| `tts` | text-to-speech, SpeechT5, single speaker | `audio` + `text` | minus spectrogram loss |

Common column names (`instruction`/`output`, `question`/`answer`, ...) are recognised; the agent maps the rest.
Data is a `.jsonl`, `.json`, `.csv` or `.parquet` file, or a Hugging Face dataset as `hf:org/name[:config[:split]]`.
Image and audio columns hold file paths or URLs.

Train on Kaggle's free GPUs (nothing heavy is installed or run on your machine):

```bash
uv run llm_factory.py chats.jsonl --goal "answer support questions in our tone" --kaggle run.ipynb
```

On kaggle.com choose Create > Notebook > File > Import notebook, set the accelerator to GPU T4 x2, turn Internet on
and run all. The winner appears as a zip in the notebook's output. Data files up to 5 MB for `text` and `decision`
are embedded in the notebook; larger files, images and audio go in as a Kaggle dataset.

Train on this machine (uses a GPU when there is one; without one only very small models are practical):

```bash
uv sync --extra llm
uv run --extra llm llm_factory.py chats.jsonl --goal "answer support questions in our tone"
```

| Option | Meaning |
|---|---|
| `--goal` | what the model is for; the agent picks the kind and the candidates from it |
| `--kind` | `text`, `decision`, `vision`, `stt` or `tts`; guessed when the data makes it obvious |
| `--models` | comma-separated Hugging Face model ids to try |
| `--no-llm` | no agent: trains `--models`, or the default candidates for the kind |
| `--kaggle FILE.ipynb` | write a Kaggle notebook for this plan and stop |
| `--parallel N` | candidates trained at once (default: one per GPU, else 2) |
| `--max-rows N` | use at most this many rows |

Set `HF_TOKEN` to use gated or private models and datasets.

The download is `runs/<run_id>_llm/<winner>-model.zip`, with a `README.txt` on how to use it. Text models also
contain `model.gguf` and a `Modelfile`, converted with llama.cpp's converter (cloned on first use):

```bash
ollama create my-model -f Modelfile
```

Ollama runs text-generation models only, so decision, vision and speech models are exported as Hugging Face model
folders. A decision model answers from the command line with `python llm_train.py decide . "some text"`.

## Tests

```bash
uv run test_ml_factory.py
```
