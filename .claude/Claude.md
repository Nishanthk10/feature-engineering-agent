# Claude.md — v1.0 · FROZEN · 2026-03-22

---

## 1. System Intent

This system is an autonomous feature engineering agent. It accepts a tabular CSV dataset and a target column name, then iteratively hypothesises, generates, evaluates, and refines engineered features using an LLM reasoning loop — stopping when AUC converges or the iteration cap is reached. It does not perform model selection, hyperparameter tuning, data ingestion, or model deployment. Success means measurable AUC lift over the raw baseline with a complete reasoning trace explaining every feature decision.

---

## 2. Hard Invariants

INVARIANT: The original input dataset must never be modified. All feature engineering operates on a deepcopy. The source CSV on disk is read-only for the duration of the agent run. This is never negotiable.

INVARIANT: No generated feature may encode the target variable directly or through a derivation that would not be available at prediction time. The leakage detector must run on every FeatureCandidate before it enters evaluation. This is never negotiable.

INVARIANT: LLM-generated Python code must only execute inside the subprocess sandbox. No generated code may run in the main process under any circumstances. This is never negotiable.

INVARIANT: The agent must terminate after at most 10 iterations, regardless of whether AUC is still improving. Early stop also triggers if AUC delta < 0.001 for 2 consecutive iterations. This is never negotiable.

INVARIANT: Every iteration must produce a complete IterationRecord written to trace.json before the next iteration begins. A partial trace is not acceptable. Write to a .tmp file first, then rename atomically. This is never negotiable.

INVARIANT: Given the same feature set, the evaluation tool must return the same AUC and SHAP values on every call. LightGBM random_state=42, train_test_split random_state=42, always. This is never negotiable.

INVARIANT: Every FeatureCandidate must have: a non-empty name, non-empty transformation_code, non-empty hypothesis, and a float SHAP value. A feature missing any of these fields must not enter the final output. Pydantic validation enforces this — do not bypass validators. This is never negotiable.

INVARIANT: The four MCP tool signatures (profile_dataset, execute_feature_code, evaluate_features, get_shap_values) must not change their input/output schema once defined in tools/schemas.py. Schema changes require a Claude.md version bump and re-verification of all affected tasks. This is never negotiable.

INVARIANT: The agent must train and record a baseline metric (AUC for classification, RMSE for regression) on raw features before any engineered features are introduced. The baseline entry must be the first entry in trace.json. This is never negotiable.

INVARIANT: The task type (classification or regression) must be determined and locked before the baseline evaluation runs. It must not change mid-run. All tools, prompts, and metrics must use the same task type for the entire agent session. No tool or class may modify TaskType after it is set. This is never negotiable.

INVARIANT: A failure in any MLflow logging call must never stop or crash the agent run. Every mlflow.* call must be wrapped in try/except Exception. On failure, print a warning and continue. Agent exit status is determined by the JSON trace and EvaluationResult only — never by MLflow state. This is never negotiable.

---

## 3. Scope Boundary

### Files CC may create:
- `agent/data_loader.py`
- `agent/loop.py`
- `agent/llm_reasoner.py`
- `agent/leakage_detector.py`
- `agent/output_formatter.py`
- `tools/profile.py`
- `tools/execute.py`
- `tools/evaluate.py`
- `tools/shap_tool.py`
- `tools/schemas.py`
- `tools/sandbox_runner.py`
- `tools/mcp_server.py`
- `api/main.py`
- `static/index.html`
- `data/generate_synthetic.py`
- `run_agent.py`
- `run_benchmark.py`
- `tests/test_scaffold.py`
- `tests/test_data_loader.py`
- `tests/test_evaluate.py`
- `tests/test_profile.py`
- `tests/test_loop.py`
- `tests/test_execute.py`
- `tests/test_shap_tool.py`
- `tests/test_leakage_detector.py`
- `tests/test_output_formatter.py`
- `tests/test_mcp_server.py`
- `tests/test_api.py`
- `tests/test_benchmark.py`
- `tests/test_regression.py`
- `tests/test_mlflow.py`
- `tests/test_e2e.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `ARCHITECTURE.md`
- `INVARIANTS.md`
- `EXECUTION_PLAN.md`

### Files CC must not modify:
- `Claude.md` — frozen, never edited inline
- Any file outside the above list without explicit task instruction
- Any file in `data/` except `generate_synthetic.py` and `.gitkeep`
- Any file in `outputs/` — these are runtime artifacts, not source files

### Conflict rule:
If a task prompt conflicts with an invariant, the invariant wins. Flag the conflict immediately. Never resolve it silently by proceeding with the task prompt.

### Scope rule:
If something is not in the task prompt, do the minimum and flag the gap. Never fill gaps with judgment calls. Never add features, endpoints, or files not specified in the task.

---

## 4. Fixed Stack

| Component | Technology | Version / Notes |
|---|---|---|
| Language | Python | 3.11+ |
| LLM | Claude claude-sonnet-4-20250514 | via `anthropic` SDK |
| LLM client env var | `ANTHROPIC_API_KEY` | loaded from `.env` via python-dotenv |
| ML model (classification) | LightGBM | `LGBMClassifier`, n_estimators=50, max_depth=4, random_state=42, class_weight="balanced" |
| ML model (regression) | LightGBM | `LGBMRegressor`, n_estimators=50, max_depth=4, random_state=42 |
| Task type | `TaskType` enum | `classification` or `regression`. Auto-detected or via `--task-type` flag. Locked before loop starts |
| Primary metric | AUC (classification) / RMSE (regression) | Stored as `primary_metric` in EvaluationResult |
| Secondary metric | F1 (classification) / R² (regression) | Stored as `secondary_metric` in EvaluationResult |
| SHAP | shap | `TreeExplainer` only — works for both task types |
| Data manipulation | pandas | DataFrame operations only |
| Numerical | numpy | array operations |
| Stats | scipy.stats | available in sandbox whitelist |
| Preprocessing | sklearn.preprocessing | available in sandbox whitelist |
| Leakage — classification | sklearn.feature_selection | `mutual_info_classif` |
| Leakage — regression | sklearn.feature_selection | `mutual_info_regression` |
| Validation | pydantic | v2 — use `model_validator` not `validator` |
| MCP | fastmcp | `FastMCP()` pattern |
| API | FastAPI + uvicorn | `api/main.py`, port 8000 |
| Experiment tracking | mlflow | local file-based, `mlflow.set_tracking_uri("./mlruns")`. All calls non-blocking |
| Testing | pytest | `pytest.mark.e2e` for end-to-end tests |
| Sandbox | subprocess | timeout=30, /tmp/fe_sandbox/ working dir |
| Trace format | JSON | list of dicts, written atomically via .tmp rename |
| Train/test split | sklearn | `train_test_split`, test_size=0.2, random_state=42, stratify=target (classification only) |
| Frontend | Vanilla HTML + JS | No React, no external CSS frameworks, no npm |
| Environment | python-dotenv | load `.env` at startup in `run_agent.py` and `api/main.py` |
| Demo datasets | synthetic_churn.csv (classification), synthetic_regression.csv (regression) | Generated by `data/generate_synthetic.py` |

### Import whitelist for sandbox_runner.py (enforced at runtime):
`pandas`, `numpy`, `scipy.stats`, `sklearn.preprocessing`, `math`, `datetime`

Any import not on this list must cause the sandbox to exit with code 1 and an error message.
