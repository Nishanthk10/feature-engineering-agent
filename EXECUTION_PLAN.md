# EXECUTION_PLAN.md — Autonomous Feature Engineering Agent

---

## Resolved Decisions

| Open Question | Resolution |
|---|---|
| Which LLM? | Claude claude-sonnet-4-20250514 via Anthropic API |
| Max iterations? | Hard cap 10; early stop if metric delta < 0.001 for 2 consecutive iterations |
| Classification only or regression? | Both. Auto-detected from target column (>20 unique numeric values = regression). Override via `--task-type` flag |
| MLflow in scope? | Yes. Local file-based (`./mlruns`). Non-blocking — all calls wrapped in try/except per INV-11 |
| Frontend or CLI? | FastAPI + single-page HTML UI. CLI via `python run_agent.py` also supported |
| Real or synthetic dataset? | Both. UCI Bank Marketing (classification) and synthetic regression dataset for demo |

---

## Session Overview

| Session | Name | Goal | Tasks | Est. Duration |
|---|---|---|---|---|
| S1 | Scaffold + Data Foundation | Repo runs, dataset loads, baseline eval works | 4 | Day 1 |
| S2 | Agent Core + Tools | LLM loop calls tools, sandbox executes code, trace logs write | 4 | Day 2–3 |
| S3 | Guardrails + Evaluation | Leakage detection, invariant enforcement, benchmark eval | 3 | Day 4 |
| S4 | MCP Exposure + UI | Tools exposed as MCP servers, FastAPI + HTML UI working | 3 | Day 5 |
| S5 | Regression + MLflow + Observability + Hardening | Regression support live, MLflow logging live, trace viewer, README, e2e test | 5 | Day 6–7 |

---

## Session 1 — Scaffold + Data Foundation

**Session goal:** A clean repo with dependencies installed, a dataset loader that validates input, and a working baseline evaluation that trains LightGBM on raw features and returns AUC + SHAP values. All verified with shell commands.

**Integration check:**
```bash
python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 0
# Must print: "Baseline AUC: [value]" and write trace.json with one entry (status: baseline)
```

---

### Task 1.1 — Repository scaffold

**Description:** Initialise repo structure, install all dependencies, confirm environment runs. No ML code yet.

**CC prompt:**
```
Create the following repository structure for a Python project called feature-agent:

feature-agent/
  run_agent.py          # CLI entry point, stub only — prints "Agent ready"
  agent/
    __init__.py
    loop.py             # stub AgentLoop class with run() method that returns None
  tools/
    __init__.py
    schemas.py          # stub file, empty for now
    profile.py          # stub ProfileTool class
    execute.py          # stub ExecuteTool class
    evaluate.py         # stub EvaluateTool class
    shap_tool.py        # stub ShapTool class
  data/                 # empty directory, add .gitkeep
  outputs/              # empty directory, add .gitkeep
  tests/
    __init__.py
    test_scaffold.py    # one test: assert run_agent.py can be imported without error
  requirements.txt      # with: fastapi, uvicorn, anthropic, lightgbm, shap, pandas, 
                        #   numpy, scipy, scikit-learn, pydantic, fastmcp, pytest, python-multipart
  .env.example          # ANTHROPIC_API_KEY=your_key_here
  README.md             # title + one-line description only

Do not implement any logic beyond stubs. Do not create any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | `pip install -r requirements.txt` runs | Zero errors, all packages install |
| TC-2 | `python run_agent.py` runs | Prints "Agent ready", exits 0 |
| TC-3 | `pytest tests/test_scaffold.py` | 1 passed |

**Verification command:**
```bash
pip install -r requirements.txt && python run_agent.py && pytest tests/test_scaffold.py -v
```

**Invariant flag:** None — scaffold only.

---

### Task 1.2 — Dataset loader and validator

**Description:** Build a `DatasetLoader` class that reads a CSV, validates the target column exists, deepcopies the DataFrame before returning it, and rejects datasets with >500MB file size or fewer than 100 rows.

**CC prompt:**
```
In agent/data_loader.py, implement a DatasetLoader class with one public method:

  load(csv_path: str, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]
    Returns: (original_df_copy, working_df_copy)
    Both are independent deep copies of the loaded data.
    Raises ValueError if:
      - File does not exist
      - target_column not in columns
      - File size > 500MB
      - Fewer than 100 rows after loading
    Must never modify the source file.

In tests/test_data_loader.py, write tests covering:
  - Happy path: valid CSV with named target column returns two independent DataFrames
  - Missing target column raises ValueError
  - File not found raises ValueError
  - Modifying the returned working_df does not affect the original_df (independence check)

Use pandas for CSV loading. Use copy.deepcopy for both return values.
Do not create any other files. Do not modify run_agent.py yet.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Valid CSV, valid target | Returns two independent DataFrames |
| TC-2 | Target column missing | ValueError raised |
| TC-3 | File not found | ValueError raised |
| TC-4 | Mutate working_df, check original_df | Original unchanged |

**Verification command:**
```bash
pytest tests/test_data_loader.py -v
```

**Invariant flag:** INV-01 (dataset immutability). Code review must confirm: deepcopy used on both return values, no write operation on source path.

---

### Task 1.3 — Baseline evaluator

**Description:** Build an `EvaluateTool` that trains LightGBM with fixed hyperparameters on a given DataFrame + target column and returns AUC, F1, and per-feature SHAP values. Fixed random seeds throughout.

**CC prompt:**
```
In tools/evaluate.py, implement EvaluateTool with one public method:

  evaluate(df: pd.DataFrame, target_col: str) -> EvaluationResult

EvaluationResult is a Pydantic model (define in tools/schemas.py) with fields:
  - auc: float
  - f1: float  
  - shap_values: dict[str, float]  # feature_name -> mean absolute SHAP value
  - feature_names: list[str]

Implementation rules:
  - Train LightGBM LGBMClassifier with: n_estimators=50, max_depth=4, random_state=42
  - Train/test split: 80/20, random_state=42, stratified on target
  - Use class_weight="balanced" to handle imbalanced datasets
  - Compute SHAP values using shap.TreeExplainer on the test split
  - If dataset > 50,000 rows, sample 5,000 rows for SHAP computation only (not for training)
  - Drop target column from features before training

In tests/test_evaluate.py, test:
  - Returns EvaluationResult with all fields populated
  - AUC is between 0.5 and 1.0 on a clean synthetic dataset
  - SHAP values dict has one key per feature column
  - Same inputs always return same AUC (determinism check — run twice, compare)

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Clean synthetic binary dataset | EvaluationResult returned, AUC 0.5–1.0 |
| TC-2 | Run twice with same inputs | AUC identical both runs |
| TC-3 | SHAP values dict | One key per feature, values are floats |
| TC-4 | Target column in DataFrame | Dropped from features, not in shap_values |

**Verification command:**
```bash
pytest tests/test_evaluate.py -v
```

**Invariant flag:** INV-06 (evaluation determinism), INV-09 (baseline always first). Code review: confirm `random_state=42` on both LGBMClassifier and train_test_split.

---

### Task 1.4 — Dataset profiler and CLI wiring

**Description:** Build `ProfileTool` that returns summary statistics for a DataFrame. Wire `run_agent.py` to accept `--dataset`, `--target`, `--max-iter` args, load data, run baseline evaluation, and write a minimal trace JSON.

**CC prompt:**
```
1. In tools/profile.py, implement ProfileTool with method:
   profile(df: pd.DataFrame, target_col: str) -> DatasetProfile
   
   DatasetProfile is a Pydantic model (add to tools/schemas.py) with:
     - row_count: int
     - column_count: int
     - target_col: str
     - feature_cols: list[str]
     - missing_rate: dict[str, float]  # col_name -> fraction missing
     - dtypes: dict[str, str]          # col_name -> dtype string

2. Update run_agent.py to:
   - Accept CLI args: --dataset (path), --target (column name), --max-iter (int, default 5)
   - Load dataset using DatasetLoader
   - Run ProfileTool to get DatasetProfile
   - Run EvaluateTool on raw features to get baseline EvaluationResult
   - Write outputs/trace.json with one entry:
     {
       "iteration": 0,
       "status": "baseline",
       "auc": <baseline_auc>,
       "f1": <baseline_f1>,
       "features_used": <list of feature names>
     }
   - Print to console: "Baseline AUC: {auc:.4f}"
   - Exit 0

3. In tests/test_profile.py:
   - Test DatasetProfile has correct row_count and column_count
   - Test target_col not in feature_cols
   - Test missing_rate keys match all non-target columns

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Run with valid dataset + target | Prints "Baseline AUC: X", exits 0 |
| TC-2 | trace.json written | Contains iteration 0, status: baseline, auc value |
| TC-3 | --max-iter 0 flag | Stops after baseline, no iteration 1 |
| TC-4 | Profile missing_rate | Correct fractions for known-missing synthetic data |

**Verification command:**
```bash
pytest tests/test_profile.py -v && python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 0 && cat outputs/trace.json
```

**Invariant flag:** INV-05 (trace completeness), INV-09 (baseline first).

---

## Session 2 — Agent Core + Tools

**Session goal:** The full agent loop runs: LLM forms a hypothesis, calls execute_feature_code tool to generate and run a pandas transformation, calls evaluate_features to measure AUC lift, reads SHAP values, decides keep/discard/mutate, and logs the IterationRecord. At least 2 full iterations complete without error.

**Integration check:**
```bash
python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 2
# Must complete 2 iterations, write 3 entries to trace.json (baseline + 2 iterations),
# print AUC for each iteration, exit 0
```

---

### Task 2.1 — Code execution sandbox

**Description:** Build `ExecuteTool` that accepts a pandas transformation code string and a DataFrame, runs it in a subprocess with import whitelist and timeout, and returns the resulting DataFrame with new columns added.

**CC prompt:**
```
In tools/execute.py, implement ExecuteTool with method:
  execute(df: pd.DataFrame, code: str) -> ExecuteResult

ExecuteResult is a Pydantic model (add to tools/schemas.py):
  - success: bool
  - new_columns: list[str]   # column names added by the code
  - error_message: str | None
  - output_df: pd.DataFrame | None  # None if success=False

Implementation rules:
  - Serialise df to /tmp/fe_sandbox/input.pkl before subprocess
  - Subprocess receives code as a string argument
  - Subprocess script (tools/sandbox_runner.py — create this file):
    - Reads input.pkl
    - Enforces import whitelist at top: only allow pandas, numpy, scipy.stats, 
      sklearn.preprocessing, math, datetime
    - exec()s the code string against {"df": loaded_df}
    - After exec, identifies new columns (columns in df now that were not before)
    - Writes result df to /tmp/fe_sandbox/output.pkl
    - Exits 0 on success, 1 on any exception
  - Main process reads output.pkl if exit code 0, parses stderr if exit code 1
  - Subprocess timeout: 30 seconds (use subprocess.run with timeout=30)
  - os.makedirs("/tmp/fe_sandbox/", exist_ok=True) before any operation

In tests/test_execute.py:
  - Happy path: code adds a valid new column, success=True, new_columns contains it
  - Invalid Python syntax: success=False, error_message not None
  - Timeout: code with time.sleep(35) returns success=False within 35 seconds
  - Disallowed import (e.g. import os): success=False

Do not use exec() in the main process. Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Valid transformation code | success=True, new column in output_df |
| TC-2 | Syntax error in code | success=False, error_message set |
| TC-3 | Disallowed import (import os) | success=False |
| TC-4 | Code runs for 35 seconds | success=False, returns within 35s |

**Verification command:**
```bash
pytest tests/test_execute.py -v
```

**Invariant flag:** INV-03 (code execution isolation). Code review: confirm no `exec()` in main process, subprocess timeout enforced, import whitelist in sandbox_runner.py.

---

### Task 2.2 — SHAP tool

**Description:** Build `ShapTool` that accepts an EvaluationResult and returns a ranked list of features by SHAP value with directional signal, formatted for LLM consumption.

**CC prompt:**
```
In tools/shap_tool.py, implement ShapTool with method:
  format_for_llm(eval_result: EvaluationResult) -> ShapSummary

ShapSummary is a Pydantic model (add to tools/schemas.py):
  - ranked_features: list[FeatureShapEntry]
  - top_3_summary: str  # human-readable string for LLM prompt injection

FeatureShapEntry (add to tools/schemas.py):
  - feature_name: str
  - mean_abs_shap: float
  - rank: int  # 1 = most important

top_3_summary must be a plain English string like:
  "Top features by importance: (1) balance_to_income_ratio (shap=0.142), 
   (2) contact_recency_score (shap=0.098), (3) age (shap=0.071)"

In tests/test_shap_tool.py:
  - Test ranked_features is sorted descending by mean_abs_shap
  - Test rank field is correct (1-indexed)
  - Test top_3_summary string contains top 3 feature names
  - Test with single feature: ranked_features has one entry, rank=1

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | EvaluationResult with 5 features | ranked_features sorted by shap descending |
| TC-2 | rank field | 1-indexed, matches sort order |
| TC-3 | top_3_summary | Contains top 3 feature names as strings |
| TC-4 | Single feature input | One entry, rank=1 |

**Verification command:**
```bash
pytest tests/test_shap_tool.py -v
```

**Invariant flag:** INV-07 (feature candidate schema) — indirectly, SHAP values are required for each FeatureCandidate.

---

### Task 2.3 — LLM reasoning layer

**Description:** Build the `LLMReasoner` class that takes dataset profile + current SHAP summary + iteration history and calls the Claude API to produce a structured hypothesis + pandas transformation code. Uses tool_use via the Anthropic API.

**CC prompt:**
```
In agent/llm_reasoner.py, implement LLMReasoner with method:
  reason(
    profile: DatasetProfile,
    shap_summary: ShapSummary,
    iteration_history: list[IterationRecord],
    current_features: list[str]
  ) -> ReasoningOutput

ReasoningOutput is a Pydantic model (add to tools/schemas.py):
  - hypothesis: str          # plain English: why this feature should help
  - feature_name: str        # name for the new feature
  - transformation_code: str # valid pandas code that adds feature_name to df
  - decision_rationale: str  # why this is the next best thing to try

Use the Anthropic client (anthropic.Anthropic(), key from ANTHROPIC_API_KEY env var).
Model: claude-sonnet-4-20250514.

System prompt must include:
  - Role: "You are an expert data scientist specialising in feature engineering."
  - Instruction: always return a JSON object with exactly these keys: 
    hypothesis, feature_name, transformation_code, decision_rationale
  - Constraint: transformation_code must only use df as the variable name for the DataFrame
  - Constraint: transformation_code must add exactly one new column named feature_name to df
  - Constraint: allowed imports in code: pandas, numpy, scipy.stats, sklearn.preprocessing

User prompt must include:
  - Dataset profile summary (row_count, feature_cols, dtypes)
  - Top 3 SHAP features from shap_summary.top_3_summary
  - Iteration history: last 3 iterations only (hypothesis + AUC delta for each)
  - Instruction: "Propose the single most promising next feature to engineer."

Parse response as JSON. Validate against ReasoningOutput Pydantic model.
If JSON parsing fails: raise ValueError with the raw response included.

In tests/test_llm_reasoner.py:
  - Mock the Anthropic client
  - Test valid JSON response parses into ReasoningOutput
  - Test malformed JSON raises ValueError
  - Test that transformation_code is present and non-empty in valid response

Do not call the real API in tests. Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Valid JSON from mocked API | ReasoningOutput with all fields |
| TC-2 | Malformed JSON from API | ValueError raised |
| TC-3 | transformation_code field | Non-empty string |
| TC-4 | History > 3 iterations | Only last 3 sent to API (check mock call args) |

**Verification command:**
```bash
pytest tests/test_llm_reasoner.py -v
```

**Invariant flag:** INV-07 (hypothesis required before code).

---

### Task 2.4 — Agent loop wiring

**Description:** Wire the full agent loop in `agent/loop.py`: baseline → [hypothesis → execute → leakage check stub → evaluate → shap → decide → log] × N iterations. Update `run_agent.py` to call `AgentLoop.run()`.

**CC prompt:**
```
In agent/loop.py, implement AgentLoop with method:
  run(
    dataset_path: str,
    target_col: str,
    max_iter: int = 5
  ) -> AgentTrace

AgentTrace is a Pydantic model (add to tools/schemas.py):
  - baseline_auc: float
  - iterations: list[IterationRecord]
  - final_feature_set: list[str]
  - final_auc: float

IterationRecord is a Pydantic model (add to tools/schemas.py):
  - iteration: int
  - hypothesis: str
  - feature_name: str
  - transformation_code: str
  - auc_before: float
  - auc_after: float
  - auc_delta: float
  - shap_summary: ShapSummary
  - decision: str  # "kept" | "discarded" | "error"
  - error_message: str | None
  - status: str    # "completed" | "failed"

Loop logic:
  1. Load dataset (DatasetLoader)
  2. Run baseline EvaluateTool, store baseline_auc
  3. Write baseline entry to outputs/trace.json (INV-05: write before iteration 1 starts)
  4. For each iteration (1 to max_iter):
     a. Call LLMReasoner.reason() to get ReasoningOutput
     b. Call ExecuteTool.execute() with transformation_code
     c. If execute failed: log iteration with decision="error", continue to next iteration
     d. LEAKAGE CHECK STUB: for now, always returns False (not leaking) — 
        add a comment: # TODO: replace with real leakage detector in Task 3.1
     e. Call EvaluateTool.evaluate() on df with new feature added
     f. Call ShapTool.format_for_llm() on new EvaluationResult
     g. Decide: if auc_delta > 0, decision="kept"; else decision="discarded"
     h. If kept: add feature to working df for next iteration
     i. Write IterationRecord to outputs/trace.json atomically:
        - write to outputs/trace.tmp.json first
        - rename to outputs/trace.json
     j. Stop early if last 2 iterations both had |auc_delta| < 0.001
     k. Stop if iteration_count >= max_iter
  5. Return AgentTrace

Update run_agent.py:
  - Call AgentLoop.run(dataset, target, max_iter)
  - Print AUC for each iteration as it completes
  - Print final summary: "Final AUC: {x:.4f} (baseline: {y:.4f}, lift: {z:.4f})"

In tests/test_loop.py:
  - Mock all tools and LLMReasoner
  - Test that baseline is written before iteration 1
  - Test that early stop triggers after 2 consecutive |delta| < 0.001
  - Test that trace.json has N+1 entries after N iterations (baseline + N)
  - Test that a failed execute still writes an IterationRecord with status="failed"

Do not call real APIs or real tools in tests. Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | 2 iterations with mocked tools | trace.json has 3 entries (baseline + 2) |
| TC-2 | 2 consecutive delta < 0.001 | Early stop, loop exits |
| TC-3 | ExecuteTool returns failure | IterationRecord written with status="failed" |
| TC-4 | Baseline written first | trace.json entry 0 exists before iteration 1 starts |

**Verification command:**
```bash
pytest tests/test_loop.py -v && python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 2
```

**Invariant flag:** INV-04 (iteration cap), INV-05 (trace completeness), INV-09 (baseline first).

---

## Session 3 — Guardrails + Evaluation

**Session goal:** Leakage detection is live and blocks leaking features. FeatureCandidate Pydantic validation enforces INV-07. Agent runs end-to-end on UCI Bank Marketing and achieves measurable AUC lift over baseline.

**Integration check:**
```bash
python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 5
# Must complete 5 iterations (or early stop), final AUC > baseline AUC, 
# no leaking features in final feature set, trace.json valid JSON
```

---

### Task 3.1 — Leakage detector

**Description:** Replace the leakage check stub with a real detector. Checks correlation > 0.95, mutual information > 0.9, and name substring match against target column.

**CC prompt:**
```
In agent/leakage_detector.py, implement LeakageDetector with method:
  is_leaking(feature_series: pd.Series, target_series: pd.Series, 
             feature_name: str, target_col: str) -> LeakageResult

LeakageResult is a Pydantic model (add to tools/schemas.py):
  - is_leaking: bool
  - reason: str | None  # set if is_leaking=True

Leakage checks (any one True = leaking):
  1. Pearson correlation abs > 0.95 between feature_series and target_series
  2. Mutual information score > 0.9 (use sklearn.feature_selection.mutual_info_classif)
  3. target_col.lower() is a substring of feature_name.lower()

In agent/loop.py, replace the leakage check stub (Task 2.4 comment) with:
  leak = LeakageDetector().is_leaking(new_col_series, target_series, feature_name, target_col)
  if leak.is_leaking:
      log iteration with decision="discarded", error_message=leak.reason
      continue

In tests/test_leakage_detector.py:
  - Test: feature identical to target → is_leaking=True (correlation check)
  - Test: feature name contains target col name → is_leaking=True (name check)
  - Test: genuinely uncorrelated feature → is_leaking=False
  - Test: reason field is set when leaking

Do not modify any other files except agent/loop.py leakage stub replacement.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Feature = target values | is_leaking=True, reason set |
| TC-2 | Feature name contains target name | is_leaking=True |
| TC-3 | Uncorrelated random feature | is_leaking=False |
| TC-4 | Leaking feature in loop | IterationRecord decision="discarded" |

**Verification command:**
```bash
pytest tests/test_leakage_detector.py -v
```

**Invariant flag:** INV-02 (leakage prevention). Code review: confirm all 3 checks present, confirm loop.py calls detector before evaluate.

---

### Task 3.2 — FeatureCandidate validation and final output

**Description:** Add `FeatureCandidate` Pydantic model enforcing INV-07. Build `OutputFormatter` that produces the final ranked feature set as a human-readable report and output CSV.

**CC prompt:**
```
1. In tools/schemas.py, add FeatureCandidate Pydantic model:
   - name: str (non-empty, validated)
   - transformation_code: str (non-empty, validated)
   - hypothesis: str (non-empty, validated)
   - mean_abs_shap: float
   - auc_delta: float
   - decision: str  # "kept" | "discarded" | "error"
   All string fields: validator raises ValueError if empty string passed.

2. In agent/output_formatter.py, implement OutputFormatter with method:
   format(trace: AgentTrace, profile: DatasetProfile) -> FormattedOutput

   FormattedOutput is a Pydantic model (add to tools/schemas.py):
     - baseline_auc: float
     - final_auc: float
     - auc_lift: float
     - kept_features: list[FeatureCandidate]  # only decision="kept"
     - report_text: str  # plain English summary of what the agent found

   report_text format:
     "Agent ran {N} iterations on {dataset_name}. 
      Baseline AUC: {x:.4f}. Final AUC: {y:.4f}. Lift: {z:.4f}.
      {K} features kept:
      1. {feature_name}: {hypothesis} (SHAP contribution: {shap:.4f})"

3. Update run_agent.py to:
   - Call OutputFormatter at the end of AgentLoop.run()
   - Write outputs/final_features.csv (one row per kept feature)
   - Print report_text to console

4. In tests/test_output_formatter.py:
   - Test: feature with empty hypothesis raises ValueError on FeatureCandidate creation
   - Test: kept_features only contains decision="kept" entries
   - Test: auc_lift = final_auc - baseline_auc
   - Test: report_text contains all kept feature names

Do not modify any other files except run_agent.py.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | FeatureCandidate with empty hypothesis | ValueError raised |
| TC-2 | Mix of kept/discarded in trace | only kept in FormattedOutput.kept_features |
| TC-3 | auc_lift calculation | Equals final_auc - baseline_auc |
| TC-4 | final_features.csv written | Exists, has one row per kept feature |

**Verification command:**
```bash
pytest tests/test_output_formatter.py -v && python run_agent.py --dataset data/bank_marketing.csv --target y --max-iter 5 && cat outputs/final_features.csv
```

**Invariant flag:** INV-07 (feature candidate schema). Code review: confirm Pydantic validators reject empty strings on all three required fields.

---

### Task 3.3 — Benchmark evaluation

**Description:** Add a benchmark runner that runs the agent on UCI Bank Marketing and a synthetic churn dataset, compares AUC before/after, and writes a benchmark report. This is the evaluation record for judging.

**CC prompt:**
```
1. Create data/generate_synthetic.py — a script that generates a synthetic churn dataset 
   (1000 rows) with known engineered features hidden in interaction terms:
   - Raw columns: age, income, account_balance, city_code, num_products, 
     years_as_customer, last_contact_days, num_contacts, prev_outcome, 
     marital_status_code, education_code, job_code
   - Target: churn (binary)
   - Hidden signal: churn is strongly predicted by 
     (income / account_balance) * (1 / last_contact_days + 1) 
     plus noise — a ratio feature and a recency decay feature that the 
     agent should discover
   - Save to data/synthetic_churn.csv

2. Create run_benchmark.py:
   - Runs agent on both datasets with max_iter=5
   - For each: records baseline AUC, final AUC, lift, features discovered
   - Writes outputs/benchmark_report.md in this format:
     ## Benchmark Results
     ### UCI Bank Marketing
     - Baseline AUC: X | Final AUC: Y | Lift: Z
     - Features discovered: [list]
     ### Synthetic Churn
     - Baseline AUC: X | Final AUC: Y | Lift: Z  
     - Features discovered: [list]
     - Hidden signal found: Yes/No (check if a ratio and recency feature were kept)

3. In tests/test_benchmark.py:
   - Test generate_synthetic.py produces 1000 rows with target column "churn"
   - Test that raw baseline AUC on synthetic data is below 0.75 
     (confirming raw features alone don't capture the hidden signal)

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | generate_synthetic.py runs | 1000 rows, churn column present |
| TC-2 | Baseline AUC on synthetic | < 0.75 (hidden signal not in raw features) |
| TC-3 | benchmark_report.md written | Contains both dataset results |

**Verification command:**
```bash
python data/generate_synthetic.py && pytest tests/test_benchmark.py -v && python run_benchmark.py && cat outputs/benchmark_report.md
```

**Invariant flag:** INV-09 (baseline always first — benchmark must show baseline before lift).

---

## Session 4 — MCP Exposure + UI

**Session goal:** All four agent tools accessible as MCP servers via `fastmcp`. FastAPI serves a single-page HTML UI where a user can upload a CSV, name the target column, run the agent, and watch iterations stream in real time.

**Integration check:**
```bash
# Terminal 1: start MCP server
python -m tools.mcp_server &
# Terminal 2: start FastAPI
uvicorn api.main:app --port 8000 &
# Terminal 3: verify MCP tools list
python -c "import fastmcp; print('MCP OK')"
# Open http://localhost:8000 in browser — upload synthetic_churn.csv, target=churn, run agent
```

---

### Task 4.1 — MCP server

**Description:** Expose the four tools as MCP servers using `fastmcp`. Each tool wraps the existing implementation.

**CC prompt:**
```
Create tools/mcp_server.py using fastmcp to expose four MCP tools:

1. profile_dataset(csv_path: str, target_col: str) -> dict
   Calls DatasetLoader.load() then ProfileTool.profile(), returns DatasetProfile as dict

2. execute_feature_code(df_json: str, code: str) -> dict
   Deserialises df from JSON (pd.read_json), calls ExecuteTool.execute(), 
   returns ExecuteResult as dict (output_df serialised to JSON string)

3. evaluate_features(df_json: str, target_col: str) -> dict
   Deserialises df, calls EvaluateTool.evaluate(), returns EvaluationResult as dict

4. get_shap_values(eval_result_json: str) -> dict
   Deserialises EvaluationResult from JSON, calls ShapTool.format_for_llm(), 
   returns ShapSummary as dict

Use fastmcp.FastMCP() to register all four tools.
Add if __name__ == "__main__": mcp.run() at the bottom.

In tests/test_mcp_server.py:
  - Test each tool can be imported and called directly (bypass MCP transport)
  - Test profile_dataset returns dict with "row_count" key
  - Test execute_feature_code with valid code returns dict with "success": true

Do not modify any existing tool files. Only create tools/mcp_server.py and tests/test_mcp_server.py.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | profile_dataset called directly | Returns dict with row_count key |
| TC-2 | execute_feature_code valid code | Returns dict with success=True |
| TC-3 | evaluate_features on small df | Returns dict with auc key |
| TC-4 | get_shap_values on eval result | Returns dict with ranked_features key |

**Verification command:**
```bash
pytest tests/test_mcp_server.py -v
```

**Invariant flag:** INV-08 (MCP tool contract stability).

---

### Task 4.2 — FastAPI backend

**Description:** Build a FastAPI app with two endpoints: `POST /run` (accepts CSV + target, runs agent, returns trace) and `GET /trace` (returns current trace.json). Used by the UI for polling.

**CC prompt:**
```
Create api/main.py with FastAPI app:

POST /run
  - Accepts: multipart form with "file" (CSV upload) and "target_col" (str) and "max_iter" (int, default 5)
  - Saves uploaded file to /tmp/fe_sandbox/upload.csv
  - Calls AgentLoop.run(dataset_path, target_col, max_iter) in a background thread
  - Returns immediately: {"status": "started", "message": "Agent running"}

GET /status
  - Returns current agent status: {"status": "running" | "complete" | "idle", "iteration": N}
  - Reads from a global state dict updated by the background thread

GET /trace
  - Reads outputs/trace.json and returns its contents as JSON
  - Returns {"trace": []} if file does not exist

GET /
  - Serves static/index.html

Create static/index.html: a single-page UI with:
  - File upload input for CSV
  - Text input for target column name
  - Number input for max iterations (default 5)
  - "Run Agent" button that POSTs to /run
  - Status div that polls GET /status every 2 seconds while running
  - Results div that fetches GET /trace when status = "complete" and renders:
    - Baseline AUC vs Final AUC
    - Table of kept features: name, hypothesis, SHAP value
  - Plain HTML + vanilla JS only. No React. No external CSS frameworks.

In tests/test_api.py using FastAPI TestClient:
  - Test GET / returns 200
  - Test GET /trace returns {"trace": []} when no trace file exists
  - Test GET /status returns {"status": "idle"} on startup

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | GET / | Returns 200 |
| TC-2 | GET /trace, no file | Returns {"trace": []} |
| TC-3 | GET /status on startup | Returns {"status": "idle"} |
| TC-4 | POST /run with valid CSV | Returns {"status": "started"} |

**Verification command:**
```bash
pytest tests/test_api.py -v && uvicorn api.main:app --port 8000 &
sleep 2 && curl http://localhost:8000/status
```

**Invariant flag:** None directly — UI layer only.

---

### Task 4.3 — Streaming iteration updates

**Description:** Update the UI to show each iteration result as it completes — agent reasoning visible in real time. This is the primary demo moment.

**CC prompt:**
```
Update static/index.html to add a live iteration log:

- Add a "Live Agent Reasoning" section below the Run button
- While status = "running", poll GET /trace every 3 seconds
- For each new IterationRecord in the trace, append a card showing:
    Iteration N
    Hypothesis: [hypothesis text]
    Feature: [feature_name]
    AUC: [auc_before] → [auc_after] (delta: [auc_delta])
    Decision: [kept / discarded / error in bold]
- Cards append in order — do not clear previous cards on each poll
- When status = "complete": stop polling, show final summary above the cards

Update GET /trace in api/main.py to also return a "status" field:
  {"trace": [...], "status": "running" | "complete" | "idle"}

No changes to any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | GET /trace | Returns status field alongside trace array |
| TC-2 | UI poll during run | New cards appear without page refresh |
| TC-3 | Discarded feature | Decision shown as "discarded" |

**Verification command:**
```bash
pytest tests/test_api.py -v
# Manual: open browser, run agent on synthetic_churn.csv, watch cards appear per iteration
```

**Invariant flag:** INV-05 (trace completeness — UI reflects what trace contains).

---

## Session 5 — Regression + MLflow + Observability + Hardening

**Session goal:** Regression target support is live and verified on a synthetic regression dataset. MLflow logs every iteration non-blocking. Trace viewer renders cleanly. README complete. End-to-end test passes on both classification and regression datasets.

**Integration check:**
```bash
# Classification run
rm -f outputs/trace.json outputs/final_features.csv
python run_agent.py --dataset data/synthetic_churn.csv --target churn --max-iter 5
# Must: complete, final AUC > baseline AUC, trace.json valid, mlruns/ directory created

# Regression run
python run_agent.py --dataset data/synthetic_regression.csv --target price --task-type regression --max-iter 5
# Must: complete, final RMSE < baseline RMSE, trace.json valid
```

---

### Task 5.1 — Regression target support

**Description:** Add `TaskType` enum, auto-detection logic in `DatasetLoader`, task-type branching in `EvaluateTool` (LGBMRegressor + RMSE/R²), leakage detector update, LLM prompt update, and `--task-type` CLI flag. All existing classification behaviour must remain unchanged.

**CC prompt:**
```
Add regression target support across these files. Do not break any existing classification behaviour.

1. In tools/schemas.py, add:
   class TaskType(str, Enum):
       classification = "classification"
       regression = "regression"

2. In agent/data_loader.py, update load() to accept optional task_type: TaskType | None = None.
   If task_type is None, auto-detect:
     - If target column has > 20 unique values AND dtype is float or int: TaskType.regression
     - Otherwise: TaskType.classification
   Return task_type as a third value: tuple[pd.DataFrame, pd.DataFrame, TaskType]

3. In tools/evaluate.py, update evaluate() to accept task_type: TaskType parameter:
   - classification: LGBMClassifier, primary_metric="auc", secondary_metric="f1", 
     class_weight="balanced"
   - regression: LGBMRegressor, primary_metric="rmse", secondary_metric="r2"
     (compute RMSE as mean_squared_error(y_test, y_pred, squared=False))
   Update EvaluationResult in schemas.py:
     - rename auc -> primary_metric: float
     - rename f1 -> secondary_metric: float
     - add task_type: TaskType

4. In agent/leakage_detector.py, update is_leaking() to accept task_type: TaskType:
   - classification: use mutual_info_classif (existing)
   - regression: use mutual_info_regression from sklearn.feature_selection

5. In agent/llm_reasoner.py, update system prompt to include task_type:
   Add to system prompt: "Task type: {task_type.value}. 
   For regression: engineer features that capture non-linear relationships, 
   interaction effects, and transformations that reduce prediction error.
   For classification: engineer features that improve class separability."

6. In run_agent.py, add CLI argument:
   --task-type: optional, choices=["classification", "regression"], default=None
   Pass to DatasetLoader.load() as task_type parameter.

7. In agent/loop.py, update AgentLoop.run() to accept task_type: TaskType | None = None.
   Pass task_type through to all tools. Store in AgentTrace.task_type field.
   Update early stop condition: for regression, stop if RMSE delta < 0.001 
   (improvement = RMSE decreased).

Update AgentTrace in schemas.py to include:
  - task_type: TaskType
  - baseline_metric: float  (rename from baseline_auc)
  - final_metric: float     (rename from final_auc)

In tests/test_regression.py, write tests:
  - Auto-detection: continuous target (> 20 unique floats) → TaskType.regression
  - Auto-detection: binary target → TaskType.classification  
  - EvaluateTool with regression task returns primary_metric as RMSE (lower than naive mean predictor)
  - LeakageDetector uses mutual_info_regression for regression task
  - AgentTrace.task_type field is set correctly

Do not modify any other files. Do not change existing test files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Target with 50 unique floats, no flag | TaskType.regression auto-detected |
| TC-2 | Binary target, no flag | TaskType.classification auto-detected |
| TC-3 | `--task-type regression` flag override | TaskType.regression regardless of auto-detect |
| TC-4 | EvaluateTool regression run | primary_metric is RMSE (float > 0) |
| TC-5 | Classification still works | Existing test_evaluate.py still passes |

**Verification command:**
```bash
pytest tests/test_regression.py -v && pytest tests/test_evaluate.py -v
```

**Invariant flag:** INV-10 (task type locked before loop starts). Code review: confirm `TaskType` set in `DatasetLoader.load()` before `AgentLoop.run()` is called, confirm no tool modifies `TaskType` after it is set, confirm `AgentTrace.task_type` is populated.

---

### Task 5.2 — MLflow integration

**Description:** Add MLflow logging to the agent loop. One parent MLflow run per agent session. One child run per iteration. All `mlflow.*` calls wrapped in try/except per INV-11. JSON trace remains source of truth — MLflow is additive only.

**CC prompt:**
```
Add MLflow logging to agent/loop.py and agent/output_formatter.py. 
All MLflow calls must be wrapped in try/except Exception per INV-11.
Do not modify any other files.

In agent/loop.py, add MLflow logging:

1. Before the baseline evaluation:
   try:
     mlflow.set_tracking_uri("./mlruns")
     mlflow.set_experiment("feature-agent")
     parent_run = mlflow.start_run(run_name=f"agent-{dataset_path}-{datetime.now().isoformat()}")
   except Exception as e:
     print(f"[MLflow warning]: {e}")
     parent_run = None

2. For the baseline (iteration 0):
   try:
     if parent_run:
       with mlflow.start_run(run_name="baseline", nested=True):
         mlflow.log_metric("baseline_auc_or_rmse", baseline_result.primary_metric)
         mlflow.log_param("task_type", task_type.value)
         mlflow.log_param("feature_count", len(profile.feature_cols))
   except Exception as e:
     print(f"[MLflow warning]: {e}")

3. For each iteration, inside the existing iteration loop after IterationRecord is built:
   try:
     if parent_run:
       with mlflow.start_run(run_name=f"iteration-{iteration}", nested=True):
         mlflow.log_metric("metric_before", record.metric_before)
         mlflow.log_metric("metric_after", record.metric_after)
         mlflow.log_metric("metric_delta", record.metric_delta)
         mlflow.log_param("hypothesis", record.hypothesis[:250])
         mlflow.log_param("feature_name", record.feature_name)
         mlflow.log_param("decision", record.decision)
         mlflow.log_text(record.transformation_code, f"code_iter_{iteration}.py")
   except Exception as e:
     print(f"[MLflow warning]: {e}")

4. After the loop ends:
   try:
     if parent_run:
       mlflow.log_metric("final_metric", trace.final_metric)
       mlflow.log_metric("total_lift", trace.final_metric - trace.baseline_metric)
       mlflow.log_metric("iterations_run", len(trace.iterations))
       mlflow.end_run()
   except Exception as e:
     print(f"[MLflow warning]: {e}")

Add mlflow to requirements.txt if not already present.

In tests/test_mlflow.py:
  - Test that loop.run() completes successfully even when mlflow.start_run raises Exception
    (mock mlflow.start_run to raise Exception, run the agent, assert trace is returned)
  - Test that mlruns/ directory is created after a real run with max_iter=1
    (use a real MLflow call — do not mock)
  - Test that a failed MLflow call does not appear in the JSON trace
    (trace.json should not contain any MLflow error information)

Do not modify any existing test files. Do not modify any files other than 
agent/loop.py, requirements.txt, and create tests/test_mlflow.py.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | MLflow raises exception on start | Agent completes, trace.json written, no crash |
| TC-2 | Normal run with max_iter=1 | mlruns/ directory created |
| TC-3 | MLflow failure | Error does not appear in trace.json |
| TC-4 | All existing tests | Still pass — loop.py changes are additive only |

**Verification command:**
```bash
pytest tests/test_mlflow.py -v && pytest tests/test_loop.py -v
```

**Invariant flag:** INV-11 (MLflow non-blocking). Code review: confirm every `mlflow.*` call is inside a try/except, confirm no mlflow state affects agent exit code or trace.json contents.

---

### Task 5.3 — Trace viewer endpoint

**Description:** Add `GET /trace/view` that returns an HTML page rendering the full agent trace as a human-readable audit log. This is what you open during the demo to show judges the reasoning chain.

**CC prompt:**
```
Add GET /trace/view to api/main.py.

Returns an HTML page (string, media_type="text/html") that renders:
  - Page title: "Agent Reasoning Trace"
  - Baseline metric and task type (from iteration 0 in trace)
  - For each subsequent iteration:
    - Iteration number and status
    - Hypothesis (blockquote style)
    - Feature name and transformation code (code block)
    - Metric before → after → delta (label as AUC or RMSE based on task_type)
    - Decision (colour coded: kept=green, discarded=grey, error=red)
    - SHAP top 3 at that iteration
  - Final summary at bottom: lift achieved, features kept, task type

Use inline CSS only. No external stylesheets. No JavaScript needed — this is a static render.

In tests/test_api.py add:
  - Test GET /trace/view returns 200 and content-type text/html
  - Test with a pre-written fixture trace.json: response HTML contains "Baseline"

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | GET /trace/view | Returns 200, content-type text/html |
| TC-2 | Trace with 2 iterations | HTML contains "Iteration 1" and "Iteration 2" |
| TC-3 | Kept feature | "kept" appears in green in HTML |
| TC-4 | Regression trace | Metric label shows "RMSE" not "AUC" |

**Verification command:**
```bash
pytest tests/test_api.py -v
```

**Invariant flag:** INV-05 (trace completeness reflected in view).

---

### Task 5.4 — README and architecture diagram

**Description:** Write the submission README with setup instructions, architecture description, agent loop explanation, and demo instructions.

**CC prompt:**
```
Write README.md with exactly these sections:

## Autonomous Feature Engineering Agent
One-paragraph problem statement.

## Agent Loop
Four bullet points: Plan, Act, Observe, Adapt — one sentence each explaining 
what the LLM does at each step.

## Architecture
Short description of components. Reference ARCHITECTURE.md for full detail.
Include this ASCII architecture diagram:

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

## Setup
Exact commands:
  git clone ...
  cd feature-agent
  pip install -r requirements.txt
  cp .env.example .env
  # add ANTHROPIC_API_KEY to .env
  python data/generate_synthetic.py

## Run
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

## MCP Tools
List all four tools with one-line descriptions.

## Observability
Explain trace.json structure. Mention /trace/view endpoint. Mention MLflow UI.

## Judging Criteria Checklist
Table mapping each criterion (autonomy, decision-making, LLM reasoning, guardrails, 
MCP, evals, observability, deployment) to where it is demonstrated in the codebase.

Do not create any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | README.md exists | All 8 sections present |
| TC-2 | Both run commands in README | Classification and regression examples present |
| TC-3 | MLflow UI command in README | `mlflow ui` command present |
| TC-4 | Judging criteria table | All 8 criteria listed |

**Verification command:**
```bash
grep -c "##" README.md
# Must return 8 or more
```

**Invariant flag:** None — documentation only.

---

### Task 5.5 — End-to-end hardening

**Description:** E2E tests for both classification and regression. Synthetic regression dataset generated. Full pipeline verified clean on both task types.

**CC prompt:**
```
1. Update data/generate_synthetic.py to also generate a regression dataset:
   Add a function generate_regression_dataset() that creates data/synthetic_regression.csv:
   - 1000 rows
   - Raw columns: sqft, bedrooms, bathrooms, age_years, distance_to_center, 
     neighbourhood_code, condition_score, garage, floors
   - Target: price (continuous float)
   - Hidden signal: price is strongly driven by 
     (sqft / age_years) * condition_score + (1 / distance_to_center) * 10000 + noise
     — an area-efficiency ratio and a location decay feature the agent should discover
   Call this function at the bottom of the script alongside the existing churn generator.

2. Update run_benchmark.py to include the regression dataset:
   Run agent on synthetic_regression.csv with --task-type regression, max_iter=5.
   Add to benchmark_report.md:
     ### Synthetic Regression (House Price)
     - Baseline RMSE: X | Final RMSE: Y | Improvement: Z%
     - Features discovered: [list]
     - Hidden signal found: Yes/No

3. Create tests/test_e2e.py with two tests:

   @pytest.mark.e2e
   def test_classification_auc_lift():
     # clean state, run agent on synthetic_churn.csv, assert final_metric > baseline_metric
     # assert trace.json valid, at least 1 iteration, task_type = "classification"

   @pytest.mark.e2e  
   def test_regression_rmse_reduction():
     # clean state, run agent on synthetic_regression.csv with task_type=regression
     # assert final_metric < baseline_metric (lower RMSE = better)
     # assert trace.json valid, at least 1 iteration, task_type = "regression"

Run both e2e tests with real LLM calls — do not mock.

Do not modify any other files.
```

**Test cases:**
| Case | Scenario | Expected |
|---|---|---|
| TC-1 | Classification e2e | final AUC > baseline AUC |
| TC-2 | Regression e2e | final RMSE < baseline RMSE |
| TC-3 | Regression trace.json | task_type = "regression", metric values are RMSE |
| TC-4 | generate_synthetic.py | Creates both synthetic_churn.csv and synthetic_regression.csv |

**Verification command:**
```bash
python data/generate_synthetic.py && pytest tests/test_e2e.py -m e2e -v && python run_benchmark.py && cat outputs/benchmark_report.md
```

**Invariant flag:** All invariants — INV-10 and INV-11 specifically. Code review on e2e: confirm task_type in trace matches what was passed in, confirm no MLflow failure caused either test to fail.
