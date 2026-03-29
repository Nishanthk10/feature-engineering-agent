"""
Tests for MLflow integration in agent/loop.py.
All MLflow calls are wrapped in try/except (INV-11); these tests verify
that a failing MLflow call never kills the agent or corrupts the trace.

Strategy: patch agent.loop.mlflow directly so tests work regardless of
whether the mlflow package is fully functional in this environment.
"""
import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tools.schemas import (
    AgentTrace,
    EvaluationResult,
    ExecuteResult,
    FeatureShapEntry,
    LeakageResult,
    ReasoningOutput,
    ShapSummary,
    TaskType,
)


# ---------------------------------------------------------------------------
# Shared helpers
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
        ranked_features=[FeatureShapEntry(feature_name="feat_a", mean_abs_shap=0.10, rank=1)],
        top_3_summary="Top: feat_a (shap=0.100)",
    )


def _make_reasoning() -> ReasoningOutput:
    return ReasoningOutput(
        hypothesis="Test hypothesis",
        feature_name="new_feat",
        transformation_code="df['new_feat'] = df['feat_a'] * 2",
        decision_rationale="Test rationale",
    )


def _make_execute_success(df: pd.DataFrame) -> ExecuteResult:
    out = df.copy()
    out["new_feat"] = 0.0
    return ExecuteResult(
        success=True, new_columns=["new_feat"], error_message=None, output_df=out,
    )


def _build_tool_patches(tmp_path: pathlib.Path, base_df: pd.DataFrame) -> dict:
    """Return patches for all external tools (not MLflow) used by AgentLoop."""
    loader_mock = MagicMock()
    loader_mock.load.return_value = (base_df.copy(), base_df.copy())
    loader_mock.detect_task_type.return_value = TaskType.classification

    eval_mock = MagicMock()
    eval_mock.evaluate.side_effect = [
        _make_eval_result(0.70),  # baseline
        _make_eval_result(0.72),  # iteration 1
    ]

    shap_mock = MagicMock()
    shap_mock.format_for_llm.return_value = _make_shap_summary()

    reasoner_mock = MagicMock()
    reasoner_mock.reason.return_value = _make_reasoning()

    execute_mock = MagicMock()
    execute_mock.execute.return_value = _make_execute_success(base_df)

    leak_mock = MagicMock()
    leak_mock.is_leaking.return_value = LeakageResult(is_leaking=False, reason=None)

    return {
        "agent.loop.DatasetLoader": MagicMock(return_value=loader_mock),
        "agent.loop.EvaluateTool": MagicMock(return_value=eval_mock),
        "agent.loop.ShapTool": MagicMock(return_value=shap_mock),
        "agent.loop.ProfileTool": MagicMock(return_value=MagicMock()),
        "agent.loop.LLMReasoner": MagicMock(return_value=reasoner_mock),
        "agent.loop.ExecuteTool": MagicMock(return_value=execute_mock),
        "agent.loop.LeakageDetector": MagicMock(return_value=leak_mock),
        "agent.loop.OUTPUTS_DIR": tmp_path / "outputs",
    }


def _run_with_patches(patches: dict) -> AgentTrace:
    from contextlib import ExitStack
    from agent.loop import AgentLoop
    with ExitStack() as stack:
        for target, mock_obj in patches.items():
            stack.enter_context(patch(target, mock_obj))
        return AgentLoop().run("fake.csv", "target", max_iter=1)


def _make_failing_mlflow() -> MagicMock:
    """Return a mock mlflow module whose start_run always raises."""
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = Exception("MLflow unavailable")
    return mock_mlflow


def _make_parent_only_mlflow() -> MagicMock:
    """Return a mock mlflow where start_run succeeds once (parent run) then raises for all nested calls."""
    call_count = [0]

    def _start_run_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock()  # parent run — truthy, so parent_run is not None
        raise Exception("nested start_run failed")

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = _start_run_side_effect
    return mock_mlflow


# ---------------------------------------------------------------------------
# Test: mlflow.start_run raising must not kill the agent
# ---------------------------------------------------------------------------

class TestMLflowFailureDoesNotKillAgent:
    def test_start_run_exception_returns_agent_trace(self, tmp_path):
        """loop.run() must return an AgentTrace even when mlflow.start_run raises."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        result = _run_with_patches(patches)
        assert isinstance(result, AgentTrace)

    def test_start_run_exception_trace_has_correct_iteration_count(self, tmp_path):
        """The returned trace must contain the expected iterations despite MLflow failure."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        result = _run_with_patches(patches)
        assert len(result.iterations) == 1

    def test_start_run_exception_baseline_metric_present(self, tmp_path):
        """baseline_metric on the trace must be populated even when MLflow fails."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        result = _run_with_patches(patches)
        assert result.baseline_metric == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Test: failed MLflow call must not appear in trace.json
# ---------------------------------------------------------------------------

class TestMLflowErrorNotInTrace:
    def test_trace_json_has_no_mlflow_keys(self, tmp_path):
        """trace.json entries must contain only normal agent fields, never MLflow error data."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        _run_with_patches(patches)

        trace_path = tmp_path / "outputs" / "trace.json"
        assert trace_path.exists()
        entries = json.loads(trace_path.read_text())
        for entry in entries:
            for key in entry:
                assert "mlflow" not in key.lower()

    def test_trace_json_baseline_entry_is_well_formed(self, tmp_path):
        """Baseline entry must have the expected keys regardless of MLflow state."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        baseline = next(e for e in entries if e["status"] == "baseline")
        assert baseline["iteration"] == 0
        assert isinstance(baseline["auc"], float)
        assert isinstance(baseline["features_used"], list)

    def test_trace_json_iteration_entry_has_no_error_from_mlflow(self, tmp_path):
        """Iteration entries must not have an error_message caused by MLflow failure."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_failing_mlflow()
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        iter_entries = [e for e in entries if e.get("status") != "baseline"]
        assert len(iter_entries) == 1
        # error_message on the iteration record must be None (no leakage/exec failure)
        assert iter_entries[0]["error_message"] is None


# ---------------------------------------------------------------------------
# Test: step 4 (post-loop) MLflow block raises — trace still returned
# ---------------------------------------------------------------------------

class TestMLflowStep4Exception:
    def test_end_run_exception_agent_still_returns_trace(self, tmp_path):
        """end_run() raising in step 4 must not prevent AgentTrace from being returned."""
        mock_mlflow = MagicMock()
        # start_run returns a truthy mock on all calls so parent_run is not None
        mock_mlflow.end_run.side_effect = Exception("end_run failed")
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = mock_mlflow
        result = _run_with_patches(patches)
        assert isinstance(result, AgentTrace)

    def test_end_run_exception_trace_json_written_correctly(self, tmp_path):
        """trace.json must contain baseline + iteration entries even when step 4 raises."""
        mock_mlflow = MagicMock()
        mock_mlflow.end_run.side_effect = Exception("end_run failed")
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = mock_mlflow
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert len(entries) == 2  # baseline + 1 completed iteration
        assert entries[0]["status"] == "baseline"
        assert entries[1]["status"] == "completed"

    def test_log_metric_exception_agent_still_returns_trace(self, tmp_path):
        """log_metric() raising in step 4 must not prevent AgentTrace from being returned."""
        mock_mlflow = MagicMock()
        # log_metric raises on every call; steps 2/3 catch their own, step 4 catches its own
        mock_mlflow.log_metric.side_effect = Exception("log_metric failed")
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = mock_mlflow
        result = _run_with_patches(patches)
        assert isinstance(result, AgentTrace)


# ---------------------------------------------------------------------------
# Test: step 2/3 nested start_run raises with active parent run
# ---------------------------------------------------------------------------

class TestMLflowNestedStartRunException:
    def test_nested_start_run_exception_agent_completes(self, tmp_path):
        """Nested start_run raising (steps 2/3) with active parent must not crash the agent."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_parent_only_mlflow()
        result = _run_with_patches(patches)
        assert isinstance(result, AgentTrace)

    def test_nested_start_run_exception_trace_has_baseline_and_iteration(self, tmp_path):
        """Trace must contain a baseline entry and one completed iteration."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_parent_only_mlflow()
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        baseline = next(e for e in entries if e["status"] == "baseline")
        assert baseline["iteration"] == 0

        iter_entries = [e for e in entries if e.get("status") != "baseline"]
        assert len(iter_entries) == 1
        assert iter_entries[0]["status"] == "completed"

    def test_nested_start_run_exception_decision_is_correct(self, tmp_path):
        """The keep/discard decision must not be corrupted by a nested start_run failure."""
        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = _make_parent_only_mlflow()
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        iter_entry = next(e for e in entries if e.get("status") != "baseline")
        # eval side_effect: baseline=0.70, iter=0.72 → delta>0 → kept
        assert iter_entry["decision"] == "kept"


# ---------------------------------------------------------------------------
# Test: error iteration path MLflow block does not crash agent
# ---------------------------------------------------------------------------

def _build_error_iteration_patches(tmp_path: pathlib.Path, mock_mlflow: MagicMock) -> dict:
    """Patches where ExecuteTool.execute always fails — exercises the error iteration path."""
    base_df = _make_df()

    loader_mock = MagicMock()
    loader_mock.load.return_value = (base_df.copy(), base_df.copy())
    loader_mock.detect_task_type.return_value = TaskType.classification

    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = _make_eval_result(0.70)  # baseline only

    shap_mock = MagicMock()
    shap_mock.format_for_llm.return_value = _make_shap_summary()

    reasoner_mock = MagicMock()
    reasoner_mock.reason.return_value = _make_reasoning()

    execute_mock = MagicMock()
    execute_mock.execute.return_value = ExecuteResult(
        success=False,
        new_columns=[],
        error_message="sandbox error: blocked import",
        output_df=None,
    )

    return {
        "agent.loop.DatasetLoader": MagicMock(return_value=loader_mock),
        "agent.loop.EvaluateTool": MagicMock(return_value=eval_mock),
        "agent.loop.ShapTool": MagicMock(return_value=shap_mock),
        "agent.loop.ProfileTool": MagicMock(return_value=MagicMock()),
        "agent.loop.LLMReasoner": MagicMock(return_value=reasoner_mock),
        "agent.loop.ExecuteTool": MagicMock(return_value=execute_mock),
        "agent.loop.LeakageDetector": MagicMock(return_value=MagicMock()),
        "agent.loop.OUTPUTS_DIR": tmp_path / "outputs",
        "agent.loop.mlflow": mock_mlflow,
    }


class TestMLflowErrorIterationPath:
    def test_error_path_mlflow_exception_does_not_crash_agent(self, tmp_path):
        """MLflow nested start_run raising inside the execute-error path must not crash."""
        patches = _build_error_iteration_patches(tmp_path, _make_parent_only_mlflow())
        result = _run_with_patches(patches)
        assert isinstance(result, AgentTrace)

    def test_error_path_failed_record_written_to_trace(self, tmp_path):
        """IterationRecord with status='failed' must appear in trace.json."""
        patches = _build_error_iteration_patches(tmp_path, _make_parent_only_mlflow())
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        iter_entries = [e for e in entries if e.get("status") != "baseline"]
        assert len(iter_entries) == 1
        assert iter_entries[0]["status"] == "failed"
        assert iter_entries[0]["decision"] == "error"

    def test_error_path_error_message_preserved_in_trace(self, tmp_path):
        """The sandbox error message must survive MLflow failure and appear in trace.json."""
        patches = _build_error_iteration_patches(tmp_path, _make_parent_only_mlflow())
        _run_with_patches(patches)

        entries = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        iter_entry = next(e for e in entries if e.get("status") != "baseline")
        assert iter_entry["error_message"] == "sandbox error: blocked import"


# ---------------------------------------------------------------------------
# Test: real MLflow run creates mlruns/ directory
# ---------------------------------------------------------------------------

class TestMLflowCreatesArtifacts:
    # Marked e2e: requires real mlflow, real filesystem,
    # and clean global mlflow state. Run with:
    # pytest tests/test_mlflow.py -m e2e -v
    @pytest.mark.e2e
    def test_mlruns_directory_created(self, tmp_path):
        """A successful loop run with real MLflow must create the mlruns/ directory."""
        from unittest.mock import MagicMock

        real_mlflow = pytest.importorskip(
            "mlflow",
            reason="mlflow not importable in this environment",
        )
        # Clean up any active run left by previous tests
        real_mlflow.end_run()
        real_mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

        # Wrap real mlflow so loop.py's own set_tracking_uri("./mlruns") call
        # cannot override the tmp_path URI we set above.
        mlflow_wrapper = MagicMock(wraps=real_mlflow)
        mlflow_wrapper.set_tracking_uri.side_effect = (
            lambda uri: real_mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        )

        patches = _build_tool_patches(tmp_path, _make_df())
        patches["agent.loop.mlflow"] = mlflow_wrapper

        try:
            _run_with_patches(patches)
        finally:
            real_mlflow.end_run()
            real_mlflow.set_tracking_uri("./mlruns")

        assert (tmp_path / "mlruns").exists()
        assert (tmp_path / "mlruns").is_dir()
