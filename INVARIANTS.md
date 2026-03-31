# INVARIANTS.md — Autonomous Feature Engineering Agent

---

## INV-01: Dataset immutability
**Condition:** The original input dataset must never be modified. All feature engineering operates on a copy. The source CSV on disk is read-only for the duration of the agent run.
**Category:** Data correctness
**Why this matters:** If the agent corrupts the source data, the user loses their original dataset and cannot re-run or audit results. Silent mutation is undetectable and catastrophic.
**Enforcement points:**
- `profile_dataset` tool: load into DataFrame, immediately deepcopy before any operation
- `execute_feature_code` tool: receives a copy, never the original reference
- No tool may write to the source file path

---

## INV-02: Feature leakage prevention
**Condition:** No generated feature may encode the target variable directly or through a derivation that would not be available at prediction time.
**Category:** Data correctness
**Why this matters:** A leaking feature produces artificially inflated AUC. The model appears excellent during evaluation and fails completely in production. This is the most dangerous silent failure mode in feature engineering.
**Enforcement points:**
- Leakage detector runs on every FeatureCandidate before it enters evaluation
- Detector checks: correlation > 0.95 with target, mutual information > 0.9 with target, feature name contains target column name as substring
- Any candidate failing the leakage check is discarded and logged — never evaluated

---

## INV-03: Code execution isolation
**Condition:** LLM-generated Python code must only execute inside the subprocess sandbox. No generated code may run in the main process.
**Category:** Security
**Why this matters:** The LLM may generate code with side effects — file writes, network calls, infinite loops, or imports that crash the process. In-process execution propagates these failures to the agent runtime.
**Enforcement points:**
- `execute_feature_code` tool: always spawns a subprocess, never uses `exec()` in-process
- Subprocess has import whitelist enforced at runtime: `pandas`, `numpy`, `scipy.stats`, `sklearn.preprocessing`, `math`, `datetime`
- Subprocess timeout: 30 seconds hard kill
- Subprocess has no write access outside `/tmp/fe_sandbox/`

---

## INV-04: Iteration cap
**Condition:** The agent must terminate after at most 10 iterations, regardless of whether AUC is still improving.
**Category:** Operational
**Why this matters:** Without a hard cap, a runaway agent consumes API tokens indefinitely. In a demo context this also means the process never completes.
**Enforcement points:**
- Agent loop checks `iteration_count >= 10` before starting any new iteration
- Early stop condition also enforced: if AUC delta < 0.001 for 2 consecutive iterations, stop immediately
- Both conditions checked at the top of the loop, not inside tool calls

---

## INV-05: Trace log completeness
**Condition:** Every iteration must produce a complete IterationRecord in the JSON trace before the next iteration begins. A partial trace is not acceptable.
**Category:** Operational
**Why this matters:** The trace log is the audit record and the primary demo artifact. A missing or partial entry makes the agent's reasoning unauditable and breaks the observability story.
**Enforcement points:**
- IterationRecord is written to the trace file atomically (write to temp file, rename) at the end of each iteration
- If the iteration fails partway through, the record is written with `status: "failed"` and the error message — never silently omitted
- Agent does not start iteration N+1 until iteration N's record is confirmed written

---

## INV-06: Evaluation determinism
**Condition:** Given the same feature set, the evaluation tool must return the same AUC and SHAP values on every call.
**Category:** Data correctness
**Why this matters:** If evaluation is non-deterministic, the agent cannot reason about whether a feature genuinely improved the model or whether the change was noise. This breaks the feedback loop.
**Enforcement points:**
- LightGBM trained with `random_state=42` on every call
- Train/test split uses `random_state=42`
- SHAP TreeExplainer called on the same fixed test split
- No shuffling or resampling outside the fixed split

---

## INV-07: Feature candidate schema
**Condition:** Every FeatureCandidate must have: a name, the transformation code that produced it, the hypothesis text that motivated it, and the SHAP value from its evaluation. A feature without all four fields may not enter the final output.
**Category:** Data correctness
**Why this matters:** Features without hypotheses are the output of brute-force search, not hypothesis-driven reasoning. The hypothesis is what differentiates this agent from featuretools. Missing it silently degrades the product to the thing it claims to replace.
**Enforcement points:**
- FeatureCandidate is a validated Pydantic model — creation fails if any field is missing
- Final output render rejects any candidate not passing Pydantic validation
- LLM prompt explicitly requires hypothesis text before code generation

---

## INV-08: MCP tool contract stability
**Condition:** The four MCP tool signatures (profile_dataset, execute_feature_code, evaluate_features, get_shap_values) must not change their input/output schema once the agent loop is wired. Schema changes require a Claude.md version bump.
**Category:** Operational
**Why this matters:** The agent loop is built against these tool contracts. A schema change mid-build silently breaks the agent's ability to parse tool outputs and form the next hypothesis.
**Enforcement points:**
- Tool schemas defined as Pydantic models in `tools/schemas.py`
- Agent loop imports from `tools/schemas.py` — never inlines schema definitions
- Any schema change requires updating `tools/schemas.py`, Claude.md version, and re-verifying all tasks that call the changed tool

---

## INV-09: Baseline model always runs first
**Condition:** The agent must train and record a baseline metric (AUC for classification, RMSE for regression) on the raw features before any engineered features are introduced. All subsequent metric comparisons are relative to this baseline.
**Category:** Data correctness
**Why this matters:** Without a baseline, metric improvement cannot be measured or claimed. The demo's core claim — "agent-engineered features improved performance from X to Y" — has no foundation.
**Enforcement points:**
- Agent loop step 0 (before iteration 1): run `evaluate_features` on raw columns only
- Baseline metric stored in `AgentTrace.baseline_metric`
- Final output report always shows baseline metric alongside final metric

---

## INV-10: Task type detection is resolved before the agent loop starts
**Condition:** The task type (classification or regression) must be determined and locked before the baseline evaluation runs. It must not change mid-run. All tools, prompts, and metrics must use the same task type for the entire agent session.
**Category:** Data correctness
**Why this matters:** If task type changes mid-run, iteration N evaluates with AUC and iteration N+1 evaluates with RMSE. The delta comparison becomes meaningless — the agent would be comparing apples to oranges and could keep or discard features based on incompatible metrics.
**Enforcement points:**
- `TaskType` is determined in `DatasetLoader.load()` or from `--task-type` CLI flag — before `AgentLoop.run()` is called
- `TaskType` is passed as an immutable parameter into `AgentLoop`, `EvaluateTool`, `LeakageDetector`, and `LLMReasoner`
- No tool or class may modify `TaskType` after it is set
- `AgentTrace` stores `task_type` as a field — mismatch between stored and active task type is a fatal error

