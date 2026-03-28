# VERIFICATION_RECORD.md

**Session:** S2 — Agent Core + Tools
**Date:** 2026-03-22
**Engineer:** Nishanth

---

## Task 2.1 — Code execution sandbox

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 2

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | Valid transformation code | success=True, new column in output_df | PASS |
| TC-2 | Syntax error in code | success=False, error_message set | PASS |
| TC-3 | Disallowed import (import os) | success=False | PASS |
| TC-4 | Code runs for 35 seconds | success=False, returns within 35s | PASS |

### Prediction Statement
TC-1: valid pandas code adding a new column will return success=True with the new column in output_df
TC-2: code with a syntax error will return success=False with error_message containing the exception text
TC-3: code containing "import os" will return success=False — blocked by the import whitelist
TC-4: code with time.sleep(35) will return success=False within 35 seconds — killed by the 30 second timeout

### CC Challenge Output
- No new columns added (new_columns=[]): accepted — valid edge case, added test
- Multiple new columns added: accepted — legitimate path, added test
- output_df row count matches input: accepted — data integrity check, added test
- output_df preserves original columns: accepted — data integrity check, added test
- from X import Y disallowed form: accepted — real security gap, whitelist must block both forms, added test
- Runtime exception without import (1/0): rejected — same code path as existing KeyError test
- sandbox_runner.py in isolation: rejected — subprocess design makes direct import impractical, ExecuteTool coverage sufficient
- Timeout path in fast suite: rejected — documented as known gap in deviations

### Code Review
Invariant touched: INV-03 (code execution isolation)
- Confirm no exec() in main process
- Confirm subprocess timeout enforced at 30 seconds
- Confirm import whitelist enforced in sandbox_runner.py

### Scope Decisions


### Verification Verdict
[ Verified ] All planned cases passed
[ Verified ] CC challenge reviewed
[ Verified ] Code review complete (if invariant-touching)
[ Verified ] Scope decisions documented

**Status:** Verified

---

## Task 2.2 — SHAP tool

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 2

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | EvaluationResult with 5 features | ranked_features sorted by shap descending | PASS |
| TC-2 | rank field | 1-indexed, matches sort order | PASS |
| TC-3 | top_3_summary | Contains top 3 feature names as strings | PASS |
| TC-4 | Single feature input | One entry, rank=1 | PASS |

### Prediction Statement
TC-1: EvaluationResult with 5 features will return ranked_features sorted descending by mean_abs_shap — highest shap first
TC-2: rank field will be 1-indexed — first entry rank=1, second rank=2, matching sort order exactly
TC-3: top_3_summary string will contain the names of the top 3 features by shap value as substrings
TC-4: EvaluationResult with single feature will return ranked_features with one entry, rank=1

### CC Challenge Output
- top_3_summary with < 3 features: accepted — edge case, should not crash, added test
- top_3_summary decimal formatting: rejected — cosmetic, too brittle
- Tied SHAP values: accepted — should not raise error, added test
- All SHAP values zero: accepted — ranks still assigned, added test
- ranked_features length equals feature count: accepted — verifies nothing truncated
- feature_name matches dict key: rejected — direct assignment, covered by TC-1

### Code Review
INV-07 confirmed:
- ranked_features sorted descending by mean_abs_shap — confirmed in shap_tool.py
- rank is 1-indexed — confirmed, first entry rank=1

### Scope Decisions


### Verification Verdict
[ Verified ] All planned cases passed
[ Verified ] CC challenge reviewed
[ Verified ] Code review complete (if invariant-touching)
[ Verified ] Scope decisions documented

**Status:** Verified

---

## Task 2.3 — LLM reasoning layer

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 2

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | Valid JSON from mocked API | ReasoningOutput with all fields | PASS |
| TC-2 | Malformed JSON from API | ValueError raised | PASS |
| TC-3 | transformation_code field | Non-empty string | PASS |
| TC-4 | History > 3 iterations | Only last 3 sent to API | PASS |

### Prediction Statement
TC-1: valid JSON response from mocked API will parse into ReasoningOutput with all four fields populated
TC-2: malformed JSON from mocked API will raise ValueError containing the raw response text
TC-3: transformation_code field will be a non-empty string in a valid response
TC-4: when iteration history has more than 3 entries, only the last 3 will appear in the mock call arguments

### CC Challenge Output
- User prompt contains profile fields: rejected — prompt content too brittle
- User prompt includes top_3_summary: rejected — same reason
- Only last 3 records sent (5 record test): accepted — TC-4 not properly covered, added test
- current_features in prompt: rejected — prompt content too brittle
- Model is claude-sonnet-4-20250514: accepted — Fixed Stack enforcement verified
- System parameter content: rejected — prompt internals too brittle
- Empty iteration_history handled: accepted — real scenario on iteration 1, added test
- Missing required key raises ValidationError: accepted — distinct from JSONDecodeError, added test
- Extra keys in JSON: rejected — Pydantic v2 library guarantee


### Code Review
INV-07 confirmed:
- hypothesis field required in ReasoningOutput — Pydantic validation enforces non-empty, confirmed in schemas.py
- System prompt instructs LLM to return hypothesis before code — confirmed in llm_reasoner.py system prompt text

### Scope Decisions
IterationRecord expanded from 3 fields to 11 in tools/schemas.py — required to avoid conflict with Task 2.4. test_llm_reasoner.py fixture updated to match. Option 1 chosen over optional fields to preserve INV-07 Pydantic enforcement.

### Verification Verdict
[ Verified ] All planned cases passed
[ Verified ] CC challenge reviewed
[ Verified ] Code review complete (if invariant-touching)
[ Verified ] Scope decisions documented

**Status:** Verified

---

## Task 2.4 — Agent loop wiring

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 2

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | 2 iterations with mocked tools | trace.json has 3 entries (baseline + 2) | PASS |
| TC-2 | 2 consecutive delta < 0.001 | Early stop, loop exits | PASS |
| TC-3 | ExecuteTool returns failure | IterationRecord written with status="failed" | PASS |
| TC-4 | Baseline written first | trace.json entry 0 exists before iteration 1 starts | PASS |

### Prediction Statement
TC-1: running 2 iterations with all tools mocked will produce trace.json with exactly 3 entries — baseline plus 2 iterations
TC-2: if 2 consecutive iterations both have auc_delta < 0.001 the loop will exit before reaching max_iter
TC-3: if ExecuteTool returns success=False the IterationRecord will be written with status="failed" and loop continues 
      to next iteration
TC-4: trace.json will contain a baseline entry as the first entry before any iteration entry appears

### CC Challenge Output
- decision="kept" updates working_df: accepted — core loop mechanic, added test
- decision="discarded" does not update working_df: accepted — converse, added test
- final_feature_set reflects kept features: accepted — AgentTrace output correctness
- max_iter=50 clamped to 10: accepted — INV-04 must be exercised
- sequential iteration numbering: accepted — trace data integrity
- tmp file absent after run: accepted — INV-05 atomic write completion
- final_auc equals last kept AUC: rejected — covered by kept/discarded tests
- auc_before matches previous auc_after: rejected — implementation internals
- final summary line printed: rejected — cosmetic output
- CLI max_iter forwarding: rejected — covered in Task 1.4 e2e tests

### Code Review
Invariants touched: INV-04, INV-05, INV-09
- INV-04: hard cap effective_max = min(max_iter, 10) confirmed on line 68
- INV-04: early stop on 2 consecutive deltas < 0.001 confirmed
- INV-05: atomic write uses Path.replace() confirmed
- INV-05: baseline entry written before loop starts confirmed
- INV-09: EvaluateTool called on raw features before any iteration confirmed

### Scope Decisions


### Verification Verdict
[ Verified ] All planned cases passed
[ Verified ] CC challenge reviewed
[ Verified ] Code review complete (if invariant-touching)
[ Verified ] Scope decisions documented

**Status:** Verified
