# SESSION_LOG.md

## Session: S2 — Agent Core + Tools
**Date started:** 2026-03-22
**Engineer:** Nishanth
**Branch:** session/s02_agent_core
**Claude.md version:** v1.0
**Status:** In Progress

## Tasks

| Task Id | Task Name | Status | Commit |
|---------|-----------|--------|--------|
| 2.1 | Code execution sandbox | Completed | f9d19f6 |
| 2.2 | SHAP tool | Completed | 2ecc7fb |
| 2.3 | LLM reasoning layer | Completed | 15e7d1e |
| 2.4 | Agent loop wiring | Completed | 637e25b |

---

## Decision Log

| Task | Decision made | Rationale |
|------|---------------|-----------|
| 2.1 | Accepted 5 CC Challenge cases: no new cols, multiple cols, row count, preserve cols, from X import Y form | Security and data integrity coverage |
| 2.1 | Rejected 3 CC Challenge cases: runtime exception, sandbox_runner isolation, timeout in fast suite | Same code path, impractical, documented gap |
| 2.2 | Accepted 4 CC Challenge cases: fewer than 3 features, tied values, all zero, length check | Edge case and integrity coverage |
| 2.2 | Rejected 2 CC Challenge cases: decimal formatting, feature_name mapping | Too brittle; already covered |
| 2.3 | Accepted 4 CC Challenge cases: history truncation, empty history, ValidationError, model name | Direct invariant and Fixed Stack coverage |
| 2.3 | Rejected 5 CC Challenge cases: prompt content assertions, system param, extra keys | Too brittle or library guarantees |
| 2.4 | Accepted 6 CC Challenge cases: kept/discarded working_df, final_feature_set, hard cap test, sequential numbering, tmp absent | Core loop correctness and INV-04/INV-05 coverage |
| 2.4 | Rejected 5 CC Challenge cases: final_auc, auc chain, summary line, CLI forwarding | Covered elsewhere or cosmetic |



---

## Deviations

| Task | Deviation observed | Action taken |
|------|--------------------|--------------|
| 2.1 | TC-4 timeout test used wrong code — time.sleep blocked by whitelist before timeout | Replaced with busy loop test that correctly exercises TimeoutExpired path |
| 2.1 | PytestUnknownMarkWarning for slow mark — pytest.ini out of scope | Warning is cosmetic, mark works correctly for filtering. Deferred to Session 5. |
| 2.3 | IterationRecord schema expansion broke test_llm_reasoner.py fixture | Updated fixture to use full 11-field schema — Option 1, correct approach per INV-07 |
| 2.4 | TestRunAgent tests in test_profile.py broke when AgentLoop was wired — tests require live API key | Moved to @pytest.mark.e2e marker — Option 1, honest about integration test nature |
| 2.4 | Stale assertion len(trace)==1 updated to >=1 in test_trace_json_has_correct_structure | Full loop writes baseline + iterations, assertion updated to match |
| 2.4 | Atomic write temp filename corrected from trace.json.tmp to trace.tmp.json | Matches actual filename used in loop.py |
| 2.4 | Hard cap of 10 iterations missing from loop.py — only default max_iter=5 existed | Added effective_max = min(max_iter, 10) on line 68 per INV-04 |

---

## Claude.md Changes

| Change | Reason | New Claude.md version | Tasks re-verified |
|--------|--------|-----------------------|-------------------|
| IterationRecord expanded from minimal to full schema in tools/schemas.py | Task 2.4 requires full schema — gap surfaced by CC at end of Task 2.3 | v1.0 unchanged — schema addition is additive, no invariant conflict | Tasks 2.3 re-verified — all 8 still pass |
| LLM provider abstracted + tests/test_llm_client.py added to scope boundary | No Anthropic key available — Gemini default. Toggle supports 4 providers. test_llm_client.py omitted from scope boundary in first v1.2 draft — corrected | v1.2 | Tasks 2.3 re-verified after refactor |

---

## Session Completion
**Session integration check:** [ ] PASSED
**All tasks verified:** [ ] Yes
**PR raised:** [ ] Yes — PR #: session/s02_agent_core → main
**Status updated to:** 
**Engineer sign-off:**