# Autonomous Feature Engineering Agent

## Autonomous Feature Engineering Agent

Given a tabular CSV dataset and a target column, this agent autonomously engineers new features using an LLM reasoning loop - iteratively hypothesising transformations, executing them in a sandboxed subprocess, evaluating their impact via AUC or RMSE, and deciding whether to keep or discard each feature. It stops when the metric converges or the iteration cap is reached, and produces a complete reasoning trace explaining every decision.

## Agent Loop

- **Plan** — The LLM receives the dataset profile, current SHAP feature importances, and iteration history, then proposes a hypothesis and Python transformation code for a new candidate feature.
- **Act** — The transformation code is executed inside an isolated subprocess sandbox; the leakage detector then checks the new column before it enters evaluation.
- **Observe** — The evaluation tool trains a LightGBM model on the augmented dataset and returns the updated primary metric (AUC or RMSE) and new SHAP values.
- **Adapt** — The agent compares the metric delta against the baseline, records a keep/discard decision with rationale in `trace.json`, and feeds the updated feature set and SHAP summary into the next iteration.

## Architecture

The system is composed of five layers: data ingestion (`DatasetLoader`), baseline evaluation (`EvaluateTool`), the LLM reasoning loop (`AgentLoop` + `LLMReasoner`), MCP-exposed tools, and output formatting (`OutputFormatter`). See [ARCHITECTURE.md](ARCHITECTURE.md) for full component detail.

```
  [CSV + target column]
         |
   [DatasetLoader + TaskType detection]
         |
   [Baseline Eval] ──> trace.json + MLflow
         |
   ┌─────▼──────────────────────────────┐
   │         Agent Loop (LLM)           │
   │  Plan → Act → Observe → Adapt      │
   │                                    │
   │  Tools (MCP):                      │
   │  profile_dataset                   │
   │  execute_feature_code              │
   │  evaluate_features                 │
   │  get_shap_values                   │
   │                                    │
   │  Guardrails:                       │
   │  leakage_detector, sandbox         │
   └─────────────────────────────────────┘
         |
   [OutputFormatter]
         |
   [final_features.csv + trace.json + MLflow UI]
```

## Setup

```bash
git clone ...
cd feature-agent
pip install -r requirements.txt
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
python data/generate_synthetic.py
```

## Run

```bash
# CLI — classification (auto-detected)
python run_agent.py --dataset data/synthetic_churn.csv --target churn --max-iter 5

# CLI — regression (explicit)
python run_agent.py --dataset data/synthetic_regression.csv --target price --task-type regression --max-iter 5

# UI
uvicorn api.main:app --port 8000
# open http://localhost:8000

# MLflow UI
mlflow ui
# open http://localhost:5000

# Benchmark
python run_benchmark.py
```

## MCP Tools

| Tool | Description |
|---|---|
| `profile_dataset` | Returns column types, missing value counts, cardinality, and summary statistics for the working dataset. |
| `execute_feature_code` | Runs LLM-generated Python transformation code inside the subprocess sandbox and returns the augmented DataFrame. |
| `evaluate_features` | Trains a LightGBM model on the current feature set and returns AUC/RMSE, F1/R², and raw SHAP values. |
| `get_shap_values` | Formats raw SHAP values into a ranked summary suitable for inclusion in the LLM reasoning prompt. |

## Observability

`outputs/trace.json` is a JSON array written atomically after every iteration. Each entry is either a baseline record (`"status": "baseline"`) containing the task type and raw metric, or an iteration record (`"status": "completed"` or `"failed"`) containing the hypothesis, transformation code, metric before/after, SHAP summary, keep/discard decision, and any error message.

The `/trace/view` endpoint (served by the FastAPI app at `http://localhost:8000/trace/view`) renders the live trace as a styled HTML page — useful for monitoring a run in progress without leaving the browser.

MLflow experiment data is written to `./mlruns/` during each run. Launch `mlflow ui` and open `http://localhost:5000` to browse parent runs, nested per-iteration child runs, logged metrics (`metric_before`, `metric_after`, `metric_delta`), and artefacts (transformation code files).

## Judging Criteria Checklist

| Criterion | Where demonstrated |
|---|---|
| **Autonomy** | `agent/loop.py` — self-directed iteration loop with early-stop logic; no human input after initial CSV upload |
| **Decision-making** | `agent/loop.py` — keep/discard logic based on metric delta; leakage rejection; error recovery without crashing |
| **LLM reasoning** | `agent/llm_reasoner.py` — structured prompt with profile, SHAP context, and iteration history; JSON-validated `ReasoningOutput` |
| **Guardrails** | `agent/leakage_detector.py` (mutual information check), `tools/sandbox_runner.py` (import whitelist, subprocess isolation) |
| **MCP tools** | `tools/mcp_server.py` — four tools with stable schemas defined in `tools/schemas.py` (INV-08) |
| **Evals** | `tools/evaluate.py` — deterministic LightGBM eval (random_state=42); `run_benchmark.py` — multi-dataset benchmark with lift reporting |
| **Observability** | `outputs/trace.json` (atomic writes, INV-05); `GET /trace/view` (live HTML viewer); `mlruns/` (MLflow experiment tracking) |
| **Deployment** | `api/main.py` — FastAPI app with `POST /run`, `GET /status`, `GET /trace`, `GET /trace/view`; `static/index.html` — zero-dependency upload UI |
