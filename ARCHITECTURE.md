# ARCHITECTURE.md — Autonomous Feature Engineering Agent

---

## 1. Problem Framing

### What this system solves
Data scientists spend 60–80% of project time on feature engineering with no systematic way to explore the feature space. They rely on intuition built over years, meaning junior practitioners produce mediocre models not because of wrong algorithms but because of weak features. This system deploys an LLM-driven agent that autonomously hypothesises, generates, evaluates, and iterates on features — closing the gap between what a senior data scientist would intuit and what a practitioner can produce today.

### What this system explicitly does not solve
- Model selection or hyperparameter tuning (that is AutoML's job — this system feeds it)
- Data ingestion or ETL pipelines
- Model deployment or serving
- Real-time / streaming feature engineering
- Feature engineering for unstructured data (images, audio, raw text corpora)

### What success looks like
A user provides a CSV dataset and names the target column. The agent runs autonomously, produces a ranked feature set with plain-English hypotheses for each feature, and delivers a measurable improvement over the baseline raw-feature model — AUC lift for classification targets, RMSE reduction for regression targets. The full reasoning trace is logged so the engineer can audit every decision the agent made.

---

## 2. Key Design Decisions

### Decision 1: Agent loop implemented as LLM + tool calls, not a fixed pipeline
**What was decided:** The agent reasons via an LLM that calls tools (profile, execute, evaluate, explain). Each iteration the LLM reads tool outputs and decides what to do next — it is not following a fixed sequence.

**Rationale:** Fixed pipelines (featuretools-style) generate features without reasoning about why they should exist. The LLM reasoning layer is what enables hypothesis-driven exploration rather than brute-force search.

**Alternatives rejected:**
- Pure featuretools / tsfresh: generates hundreds of undirected features, adds noise, no hypothesis trail
- Hardcoded heuristics: cannot generalise across domains (financial data vs retail vs healthcare)

**Challenge:** LLM reasoning introduces non-determinism — the agent may make different decisions on repeated runs.
**Assessment:** Accepted. The agent's value is discovery, not reproducibility. The trace log makes decisions auditable. A random seed on the LightGBM eval layer keeps the evaluation deterministic even if the hypothesis path varies.

---

### Decision 2: Code execution in a sandboxed subprocess
**What was decided:** LLM-generated pandas transformation code runs in a subprocess with a whitelist of allowed imports, a 30-second timeout, and no filesystem write access outside a designated `/tmp/fe_sandbox/` directory.

**Rationale:** The LLM generates executable Python. Without sandboxing, a malformed or adversarial generation could corrupt the dataset or the host system.

**Alternatives rejected:**
- RestrictedPython: adds complexity with incomplete coverage of all dangerous patterns
- In-process exec(): no isolation, any exception or bad import can crash the agent

**Challenge:** Subprocess sandboxing adds latency per iteration (~200ms overhead).
**Assessment:** Accepted. Safety outweighs latency at this scale.

---

### Decision 3: LightGBM as the evaluation model
**What was decided:** Every feature set candidate is evaluated by training a LightGBM model with fixed hyperparameters (50 estimators, depth 4, 3-fold CV). AUC is the primary metric for classification; RMSE for regression.

**Rationale:** LightGBM is fast (seconds per train on datasets up to 100k rows), handles missing values natively, and produces SHAP values directly. The fixed hyperparameters eliminate hyperparameter noise — the only variable across iterations is the feature set.

**Alternatives rejected:**
- Random Forest: slower, less SHAP support
- Logistic Regression: too sensitive to feature scale, would conflate feature quality with normalisation quality
- Full AutoML per iteration: too slow (minutes per iteration)

**Challenge:** LightGBM AUC is a proxy — the final model may use a different algorithm.
**Assessment:** Accepted. The goal is relative signal (did this feature help?), not absolute model quality.

---

### Decision 4: SHAP values as the feedback signal to the LLM
**What was decided:** After each evaluation, SHAP values for each feature are computed and passed back to the LLM as the observation input for the next iteration.

**Rationale:** SHAP values tell the LLM not just whether a feature helped overall (AUC), but which features contributed, in which direction, and for which samples. This enables directed hypothesis formation rather than blind trial.

**Alternatives rejected:**
- Feature importance (permutation or gain): less granular, no directional signal
- Correlation with target: misses non-linear relationships, which is where most interesting features live

**Challenge:** SHAP computation adds time (~5–15s for 100k rows, 20 features).
**Assessment:** Accepted. This is the core intelligence signal. It cannot be removed without degrading the agent to a random search.

---

### Decision 5: MCP tool exposure
**What was decided:** The four agent tools (profile_dataset, execute_feature_code, evaluate_features, get_shap_values) are exposed as MCP servers using `fastmcp`.

**Rationale:** MCP compliance is an extra-credit judging criterion. It also means the agent tools can be driven by Claude Desktop or any MCP-compatible client — making the system extensible beyond the hackathon demo.

**Alternatives rejected:**
- Internal function calls only: no external composability, misses judging criterion

---

### Decision 6: JSON trace log as the observability artifact
**What was decided:** Every iteration is logged as a structured JSON entry: hypothesis formed, code generated, features created, metric before/after, SHAP delta per feature, keep/discard/mutate decision. Structured logs are written to `outputs/agent.log` and exposed via the `GET /logs` endpoint.

**Rationale:** The JSON trace is the demo's most powerful artifact — it makes the plan → act → observe → adapt loop visible to judges and serves as the evaluation record for correctness testing. The `agent.log` file provides per-component DEBUG logging for diagnosing failures without polluting stdout.

**Alternatives rejected:**
- Console logs only: unstructured, cannot be replayed or audited
- External experiment tracking (MLflow, W&B): adds a dependency and failure mode without improving the core observability story; JSON trace is programmatically parseable without any server running

---

### Decision 7: Regression target support
**What was decided:** The agent supports both binary classification and regression targets. Target type is auto-detected from the target column: if the column has more than 20 unique values and is numeric, it is treated as regression. Otherwise classification.

**Rationale:** Regression is the second most common ML task after classification. Supporting it doubles the range of datasets the agent can demo on and makes the system genuinely more useful. The implementation cost is low — it is a metric branching decision (AUC → RMSE) inside EvaluateTool and a prompt adjustment to the LLM system message.

**What changes vs classification only:**
- EvaluateTool: use `LGBMRegressor` instead of `LGBMClassifier`, primary metric becomes RMSE, secondary metric becomes R²
- SHAP: TreeExplainer works identically for both — no change
- LLM system prompt: include task type so hypotheses are appropriate (regression features differ from classification features)
- Leakage detector: correlation threshold still applies; mutual information uses `mutual_info_regression` instead of `mutual_info_classif`

**Challenge:** Auto-detection of target type may misclassify an ordinal classification target (e.g. ratings 1–5 with 5 unique values) as classification when regression would be more appropriate.
**Assessment:** Accepted. Add a `--task-type` CLI flag so the user can override auto-detection. Document the threshold (20 unique values) in the README.

---

## 3. Challenge My Decisions

| Decision | Strongest argument against | Assessment |
|---|---|---|
| LLM reasoning loop | Non-determinism makes the agent hard to test and results hard to reproduce | Accepted — trace log + fixed eval seed makes reasoning auditable even if not reproducible |
| Subprocess sandbox | Adds latency, subprocess isolation is imperfect on some OS configurations | Accepted — safety is non-negotiable; latency is acceptable |
| LightGBM as eval | AUC proxy may mislead on imbalanced datasets | Mitigated — add class-weight balancing and report both AUC and F1 |
| SHAP as feedback | SHAP computation on large datasets is slow | Mitigated — sample 5k rows for SHAP if dataset > 50k rows |
| MCP exposure | Adds ~1 day of build time for a judging criterion that is "extra credit" | Accepted — MCP is 1 day well spent; it also makes the architecture story stronger |
| JSON trace + structured logs | Two output formats means two things to keep in sync | Accepted — logs are append-only and independent; JSON trace is source of truth |
| Regression support | Auto-detection of target type can misclassify ordinal targets | Mitigated — `--task-type` CLI flag lets user override detection |

---

## 4. Key Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM generates syntactically invalid Python | Medium | High — iteration blocked | Sandbox catches exception, LLM retries with error message as context |
| Feature leakage (generated feature encodes target) | Medium | High — invalidates model | Leakage detector runs before any feature is accepted |
| AUC does not improve after N iterations | High | Medium — agent keeps running | Hard stop at 10 iterations or <0.001 AUC delta for 2 consecutive iterations |
| Demo dataset too large for SHAP in demo window | Medium | Medium — slow demo | Pre-sample dataset to 10k rows for demo run |
| Subprocess import whitelist too restrictive | Low | Medium — blocks valid transformations | Whitelist includes pandas, numpy, scipy.stats, sklearn.preprocessing |

---

## 5. Key Assumptions

- Input data is tabular CSV with a named target column
- Target column is binary classification or continuous regression — auto-detected, overridable via `--task-type` flag
- Dataset fits in memory (< 500MB)
- Python 3.11+ runtime available
- Claude API key available via environment variable (`ANTHROPIC_API_KEY`)

---

## 6. Open Questions

All resolved before execution planning:

| Question | Resolution |
|---|---|
| Which LLM? | Claude claude-sonnet-4-20250514 via Anthropic API |
| Max iterations? | Hard cap of 10. Stop early if metric delta < 0.001 for 2 consecutive iterations |
| Classification only or regression too? | Both. Auto-detected from target column. Override via `--task-type classification|regression` |
| Experiment tracking? | JSON trace at `outputs/trace.json` + structured log at `outputs/agent.log`. No external tracking dependency. |
| Frontend or CLI? | FastAPI + minimal HTML single-page UI. CLI also supported via `python run_agent.py` |
| Real dataset or synthetic? | Both. UCI Bank Marketing (classification) and a synthetic regression dataset for demo |

---

## 7. Future Enhancements (Parking Lot)

| Enhancement | Rationale for deferral |
|---|---|
| Time-series lag features | Requires temporal dataset structure validation and ordering guarantees; deferred to post-hackathon |
| Multi-table feature joins | Requires a schema-linking layer, join key inference, many-to-many explosion handling, and leakage detector updates for joined data — effectively a separate project; not feasible solo in one week |
| Parallel iteration (try N hypotheses simultaneously) | Requires redesigning the trace schema for concurrent writes and invalidates INV-05 as currently written; sequential iteration is sufficient for demo and the architecture is designed to make this addition clean post-hackathon |

---

## 8. Data Model

| Entity | What it represents |
|---|---|
| `Dataset` | The input CSV loaded into a pandas DataFrame. Immutable after load |
| `TaskType` | Enum: `classification` or `regression`. Auto-detected or user-specified via CLI flag |
| `FeatureCandidate` | A single engineered feature: name, transformation code, hypothesis, SHAP value, keep/discard decision |
| `IterationRecord` | One full agent loop: iteration number, hypothesis text, features attempted, metric before/after (AUC or RMSE), SHAP delta, LLM decision |
| `AgentTrace` | The complete ordered list of IterationRecords for one agent run. Written as JSON to `outputs/trace.json` |
| `EvaluationResult` | Primary metric (AUC for classification, RMSE for regression), secondary metric (F1 or R²), SHAP values per feature |
| `FormattedOutput` | Final ranked feature set with plain-English explanations. Delivered to UI and written to output CSV |
