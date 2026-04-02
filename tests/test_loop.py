"""
Tests for AgentLoop. All external tools and the LLM are mocked.
No real API calls, no real LightGBM, no real sandbox execution.
"""
import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agent.loop import AgentLoop
from tools.schemas import (
    AgentTrace,
    EvaluationResult,
    ExecuteResult,
    FeatureShapEntry,
    IterationRecord,
    LeakageResult,
    ReasoningOutput,
    ShapSummary,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "target": rng.integers(0, 2, n),
    })


def _make_eval_result(auc: float = 0.72) -> EvaluationResult:
    return EvaluationResult(
        auc=auc,
        f1=0.65,
        shap_values={"feat_a": 0.10, "feat_b": 0.07},
        feature_names=["feat_a", "feat_b"],
    )


def _make_shap_summary() -> ShapSummary:
    return ShapSummary(
        ranked_features=[
            FeatureShapEntry(feature_name="feat_a", mean_abs_shap=0.10, rank=1),
            FeatureShapEntry(feature_name="feat_b", mean_abs_shap=0.07, rank=2),
        ],
        top_3_summary="Top features by importance: (1) feat_a (shap=0.100), (2) feat_b (shap=0.070)",
    )


def _make_reasoning(n: int = 1) -> ReasoningOutput:
    return ReasoningOutput(
        hypothesis=f"Hypothesis {n}",
        feature_name=f"new_feat_{n}",
        transformation_code=f"df['new_feat_{n}'] = df['feat_a'] + {n}",
        decision_rationale="Test rationale",
    )


def _make_execute_success(df: pd.DataFrame, col: str) -> ExecuteResult:
    out = df.copy()
    out[col] = 0.0
    return ExecuteResult(
        success=True,
        new_columns=[col],
        error_message=None,
        output_df=out,
    )


def _make_execute_failure(msg: str = "SyntaxError: bad code") -> ExecuteResult:
    return ExecuteResult(
        success=False,
        new_columns=[],
        error_message=msg,
        output_df=None,
    )


def _patch_all(
    tmp_path: pathlib.Path,
    *,
    eval_aucs: list[float],       # one per EvaluateTool.evaluate() call (index 0 = baseline)
    reasoning_outputs: list[ReasoningOutput],
    execute_results: list[ExecuteResult],
):
    """Return a context-manager dict of patches targeting agent.loop's imports."""

    base_df = _make_df()

    loader_mock = MagicMock()
    loader_mock.load.return_value = (base_df.copy(), base_df.copy())

    eval_mock = MagicMock()
    eval_mock.evaluate.side_effect = [_make_eval_result(a) for a in eval_aucs]

    shap_mock = MagicMock()
    shap_mock.format_for_llm.return_value = _make_shap_summary()

    profile_mock = MagicMock()
    profile_mock.profile.return_value = MagicMock()

    reasoner_mock = MagicMock()
    reasoner_mock.reason.side_effect = reasoning_outputs

    execute_mock = MagicMock()
    execute_mock.execute.side_effect = execute_results

    patches = {
        "agent.loop.DatasetLoader": MagicMock(return_value=loader_mock),
        "agent.loop.EvaluateTool": MagicMock(return_value=eval_mock),
        "agent.loop.ShapTool": MagicMock(return_value=shap_mock),
        "agent.loop.ProfileTool": MagicMock(return_value=profile_mock),
        "agent.loop.LLMReasoner": MagicMock(return_value=reasoner_mock),
        "agent.loop.ExecuteTool": MagicMock(return_value=execute_mock),
        "agent.loop.OUTPUTS_DIR": tmp_path / "outputs",
    }
    return patches


def _apply_patches(patches: dict):
    from contextlib import ExitStack
    stack = ExitStack()
    for target, mock in patches.items():
        if target.startswith("agent.loop."):
            stack.enter_context(patch(target, mock))
    return stack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineWrittenFirst:
    def test_baseline_entry_is_first_in_trace(self, tmp_path):
        reasoning = [_make_reasoning(1)]
        exec_res = [_make_execute_success(_make_df(), "new_feat_1")]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70, 0.73],
            reasoning_outputs=reasoning,
            execute_results=exec_res,
        )
        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=1)

        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert trace[0]["iteration"] == 0
        assert trace[0]["status"] == "baseline"

    def test_baseline_auc_matches_first_evaluate_call(self, tmp_path):
        reasoning = [_make_reasoning(1)]
        exec_res = [_make_execute_success(_make_df(), "new_feat_1")]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.68, 0.71],
            reasoning_outputs=reasoning,
            execute_results=exec_res,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=1)

        assert result.baseline_auc == pytest.approx(0.68)


class TestEarlyStop:
    def test_early_stop_after_2_consecutive_small_deltas(self, tmp_path):
        # Baseline=0.700; iter 1: AUC=0.6999 → delta=-0.0001 → discarded (count=1);
        # iter 2: AUC=0.6998 → delta=-0.0002 → discarded (count=2) → early stop.
        # Negative deltas ensure decision="discarded", triggering the new early-stop rule.
        reasoning = [_make_reasoning(i) for i in range(1, 6)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, 6)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.700, 0.6999, 0.6998, 0.680, 0.670, 0.660],
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=5)

        assert len(result.iterations) == 2

    def test_early_stop_resets_on_large_delta(self, tmp_path):
        # iter 1: delta=+0.050  → KEPT (count resets to 0, current_metric=0.750);
        # iter 2: delta=-0.0001 → discarded (count=1, threshold=2 because iter<=3);
        # iter 3: delta=-0.0002 → discarded (count=2, threshold=2) → early stop.
        # All triggering iterations are <= 3 (strict phase), so threshold=2 applies.
        reasoning = [_make_reasoning(i) for i in range(1, 4)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, 4)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.700, 0.750, 0.7499, 0.7498],
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=5)

        assert len(result.iterations) == 3

    def test_kept_with_small_delta_does_not_trigger_early_stop(self, tmp_path):
        # All 3 iterations are KEPT with small positive delta — counter must reset each time,
        # so the agent runs all max_iter iterations without early stopping.
        reasoning = [_make_reasoning(i) for i in range(1, 4)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, 4)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.700, 0.7005, 0.7008, 0.7011],
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=3)

        assert len(result.iterations) == 3


class TestTraceEntryCount:
    def test_trace_has_baseline_plus_n_iterations(self, tmp_path):
        n = 3
        reasoning = [_make_reasoning(i) for i in range(1, n + 1)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, n + 1)]
        # Use large deltas so no early stop
        aucs = [0.70] + [0.70 + (i * 0.01) for i in range(1, n + 1)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=aucs,
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=n)

        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert len(trace) == n + 1  # baseline + n iterations


class TestFailedExecuteRecord:
    def test_failed_execute_writes_iteration_record(self, tmp_path):
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70],  # only baseline needed
            reasoning_outputs=[_make_reasoning(1)],
            execute_results=[_make_execute_failure("SyntaxError: bad code")],
        )
        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=1)

        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert len(trace) == 2  # baseline + 1 failed iteration
        failed = trace[1]
        assert failed["status"] == "failed"
        assert failed["decision"] == "error"
        assert failed["error_message"] is not None

    def test_failed_execute_does_not_raise(self, tmp_path):
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70],
            reasoning_outputs=[_make_reasoning(1)],
            execute_results=[_make_execute_failure()],
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=1)

        assert isinstance(result, AgentTrace)

    def test_failed_execute_auc_unchanged(self, tmp_path):
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70],
            reasoning_outputs=[_make_reasoning(1)],
            execute_results=[_make_execute_failure()],
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=1)

        assert result.final_auc == pytest.approx(0.70)


class TestWorkingDfMutation:
    def test_kept_feature_present_in_next_iteration_execute_call(self, tmp_path):
        # Iter 1: kept (AUC improves). Iter 2: execute receives df with new_feat_1.
        base_df = _make_df()
        df_with_feat1 = base_df.copy()
        df_with_feat1["new_feat_1"] = 0.0

        exec_results = [
            ExecuteResult(success=True, new_columns=["new_feat_1"], error_message=None, output_df=df_with_feat1),
            _make_execute_success(df_with_feat1, "new_feat_2"),
        ]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70, 0.75, 0.76],  # baseline, iter1 (kept), iter2
            reasoning_outputs=[_make_reasoning(1), _make_reasoning(2)],
            execute_results=exec_results,
        )
        execute_cls = MagicMock()
        execute_inst = MagicMock()
        execute_inst.execute.side_effect = exec_results
        execute_cls.return_value = execute_inst
        patches["agent.loop.ExecuteTool"] = execute_cls

        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=2)

        second_call_df = execute_inst.execute.call_args_list[1][0][0]
        assert "new_feat_1" in second_call_df.columns

    def test_discarded_feature_absent_from_next_iteration_execute_call(self, tmp_path):
        # Iter 1: discarded (AUC does not improve). Iter 2: execute receives original df.
        base_df = _make_df()
        df_with_feat1 = base_df.copy()
        df_with_feat1["new_feat_1"] = 0.0

        exec_results = [
            ExecuteResult(success=True, new_columns=["new_feat_1"], error_message=None, output_df=df_with_feat1),
            _make_execute_success(base_df, "new_feat_2"),
        ]
        execute_cls = MagicMock()
        execute_inst = MagicMock()
        execute_inst.execute.side_effect = exec_results
        execute_cls.return_value = execute_inst

        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70, 0.68, 0.72],  # iter1 AUC lower → discarded
            reasoning_outputs=[_make_reasoning(1), _make_reasoning(2)],
            execute_results=exec_results,
        )
        patches["agent.loop.ExecuteTool"] = execute_cls

        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=2)

        second_call_df = execute_inst.execute.call_args_list[1][0][0]
        assert "new_feat_1" not in second_call_df.columns

    def test_final_feature_set_contains_kept_features_only(self, tmp_path):
        # Iter 1: kept. Iter 2: discarded. final_feature_set should include new_feat_1, not new_feat_2.
        base_df = _make_df()
        df_with_feat1 = base_df.copy()
        df_with_feat1["new_feat_1"] = 0.0
        df_with_feat2 = df_with_feat1.copy()
        df_with_feat2["new_feat_2"] = 0.0

        exec_results = [
            ExecuteResult(success=True, new_columns=["new_feat_1"], error_message=None, output_df=df_with_feat1),
            ExecuteResult(success=True, new_columns=["new_feat_2"], error_message=None, output_df=df_with_feat2),
        ]
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70, 0.75, 0.73],  # iter1 kept, iter2 discarded
            reasoning_outputs=[_make_reasoning(1), _make_reasoning(2)],
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=2)

        assert "new_feat_1" in result.final_feature_set
        assert "new_feat_2" not in result.final_feature_set


class TestHardCap:
    def test_max_iter_50_clamped_to_10_iterations(self, tmp_path):
        n = 11  # more than the hard cap of 10
        reasoning = [_make_reasoning(i) for i in range(1, n + 1)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, n + 1)]
        # Large AUC gains so early stop never triggers
        aucs = [0.70 + i * 0.01 for i in range(n + 1)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=aucs,
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            result = AgentLoop().run("fake.csv", "target", max_iter=50)

        assert len(result.iterations) == 10


class TestTraceSequencing:
    def test_iteration_numbers_sequential_from_1(self, tmp_path):
        n = 3
        reasoning = [_make_reasoning(i) for i in range(1, n + 1)]
        df = _make_df()
        exec_results = [_make_execute_success(df, f"new_feat_{i}") for i in range(1, n + 1)]
        aucs = [0.70 + i * 0.01 for i in range(n + 1)]
        patches = _patch_all(
            tmp_path,
            eval_aucs=aucs,
            reasoning_outputs=reasoning,
            execute_results=exec_results,
        )
        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=n)

        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        iteration_nums = [e["iteration"] for e in trace[1:]]  # skip baseline
        assert iteration_nums == list(range(1, n + 1))

class TestLeakageDiscarded:
    def test_leaking_feature_produces_discarded_record(self, tmp_path):
        reasoning = _make_reasoning(1)
        exec_res = _make_execute_success(_make_df(), reasoning.feature_name)
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70],  # only baseline; leakage skips evaluation
            reasoning_outputs=[reasoning],
            execute_results=[exec_res],
        )
        leaking = LeakageResult(
            is_leaking=True,
            reason="Pearson correlation with target is 0.9800 (threshold: 0.95).",
        )
        with _apply_patches(patches):
            with patch("agent.loop.LeakageDetector") as mock_cls:
                mock_cls.return_value.is_leaking.return_value = leaking
                AgentLoop().run("fake.csv", "target", max_iter=1)

        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert len(trace) == 2  # baseline + 1 leakage iteration
        leaked = trace[1]
        assert leaked["decision"] == "discarded"
        assert leaked["status"] == "failed"
        assert leaked["error_message"] is not None


class TestTraceSequencing:
    def test_tmp_file_absent_after_successful_run(self, tmp_path):
        patches = _patch_all(
            tmp_path,
            eval_aucs=[0.70, 0.73],
            reasoning_outputs=[_make_reasoning(1)],
            execute_results=[_make_execute_success(_make_df(), "new_feat_1")],
        )
        with _apply_patches(patches):
            AgentLoop().run("fake.csv", "target", max_iter=1)

        assert not (tmp_path / "outputs" / "trace.tmp.json").exists()
