# VERIFICATION_RECORD.md

**Session:** S3 — Guardrails + Evaluation
**Date:** 2026-03-28
**Engineer:** Nishanth

---

## Task 3.1 — Leakage detector

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 3

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | Feature identical to target values | is_leaking=True, reason set | |
| TC-2 | Feature name contains target col name | is_leaking=True | |
| TC-3 | Uncorrelated random feature | is_leaking=False | |
| TC-4 | Leaking feature in loop | IterationRecord decision="discarded" | |

### Prediction Statement


### CC Challenge Output


### Code Review
Invariant touched: INV-02 (leakage prevention)
- Confirm all 3 checks present: correlation > 0.95, mutual info > 0.9, name substring
- Confirm loop.py calls detector before evaluate
- Confirm leaking features never enter EvaluateTool

### Scope Decisions


### Verification Verdict
[ ] All planned cases passed
[ ] CC challenge reviewed
[ ] Code review complete (if invariant-touching)
[ ] Scope decisions documented

**Status:**

---

## Task 3.2 — FeatureCandidate validation and final output

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 3

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | FeatureCandidate with empty hypothesis | ValueError raised | |
| TC-2 | Mix of kept/discarded in trace | only kept in FormattedOutput.kept_features | |
| TC-3 | auc_lift calculation | Equals final_auc - baseline_auc | |
| TC-4 | final_features.csv written | Exists, has one row per kept feature | |

### Prediction Statement


### CC Challenge Output


### Code Review
Invariant touched: INV-07 (feature candidate schema)
- Confirm Pydantic validators reject empty strings on name, transformation_code, hypothesis
- Confirm final output render rejects any candidate not passing validation

### Scope Decisions


### Verification Verdict
[ ] All planned cases passed
[ ] CC challenge reviewed
[ ] Code review complete (if invariant-touching)
[ ] Scope decisions documented

**Status:**

---

## Task 3.3 — Benchmark evaluation

### Test Cases Applied
Source: EXECUTION_PLAN.md Session 3

| Case | Scenario | Expected | Result |
|------|----------|----------|--------|
| TC-1 | generate_synthetic.py runs | 1000 rows, churn column present | |
| TC-2 | Baseline AUC on synthetic | < 0.75 (hidden signal not in raw features) | |
| TC-3 | benchmark_report.md written | Contains both dataset results | |

### Prediction Statement


### CC Challenge Output


### Code Review
Invariant touched: INV-09 (baseline always first)
- Confirm benchmark shows baseline metric before lift for both datasets

### Scope Decisions


### Verification Verdict
[ ] All planned cases passed
[ ] CC challenge reviewed
[ ] Code review complete (if invariant-touching)
[ ] Scope decisions documented

**Status:**
```

---

**3. Send this to Claude Code:**
```
I am starting Session 3 (Guardrails + Evaluation) of this project.

Re-read Claude.md from the .claude/ directory. Claude.md version 
is v1.2. All invariants and scope boundaries apply without exception.

Session goal: leakage detection is live and blocks leaking features, 
FeatureCandidate Pydantic validation enforces INV-07, agent runs 
end-to-end on synthetic dataset and achieves measurable AUC lift 
over baseline.

I will give you one task at a time. Wait for my instruction before 
starting each task. Do not build ahead. Do not create files not 
listed in the task prompt.

Ready to begin Task 3.1.
```

---

**4. Write your prediction statement for Task 3.1 before pasting the CC prompt:**
```
### Prediction Statement
TC-1: a feature series identical to the target will return 
      is_leaking=True with reason set describing the correlation check
TC-2: a feature whose name contains the target column name as a 
      substring will return is_leaking=True
TC-3: a genuinely random feature uncorrelated with the target 
      will return is_leaking=False with reason=None
TC-4: when the loop encounters a leaking feature it will write 
      an IterationRecord with decision="discarded" and continue 
      to the next iteration without calling EvaluateTool