"""
Tests for regression target support introduced in Task 5.1.
Covers: auto-detection in DatasetLoader, EvaluateTool regression path,
LeakageDetector regression MI branch, AgentTrace task_type field,
AgentLoop regression keep/discard logic, and LLMReasoner task_type note.
"""
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
SYNTHETIC_CSV = str(DATA_DIR / "synthetic_churn.csv")


def _regression_csv(n: int = 200) -> str:
    """Write a CSV with a continuous target (many unique floats) and return its path."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "target": rng.standard_normal(n) * 10 + 50,  # continuous, >20 unique values
    })
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _binary_csv(n: int = 200) -> str:
    """Write a CSV with a binary target and return its path."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "label": rng.integers(0, 2, n),
    })
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# TaskType enum
# ---------------------------------------------------------------------------

class TestTaskTypeEnum:
    def test_classification_value(self):
        from tools.schemas import TaskType
        assert TaskType.classification.value == "classification"

    def test_regression_value(self):
        from tools.schemas import TaskType
        assert TaskType.regression.value == "regression"

    def test_is_string_subclass(self):
        from tools.schemas import TaskType
        assert isinstance(TaskType.classification, str)


# ---------------------------------------------------------------------------
# DatasetLoader auto-detection
# ---------------------------------------------------------------------------

class TestDatasetLoaderAutoDetect:
    def test_continuous_target_detected_as_regression(self):
        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        csv_path = _regression_csv()
        detected = DatasetLoader().detect_task_type(csv_path, "target")
        assert detected == TaskType.regression

    def test_binary_target_detected_as_classification(self):
        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        csv_path = _binary_csv()
        detected = DatasetLoader().detect_task_type(csv_path, "label")
        assert detected == TaskType.classification

    def test_churn_csv_detected_as_classification(self):
        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        detected = DatasetLoader().detect_task_type(SYNTHETIC_CSV, "churn")
        assert detected == TaskType.classification

    def test_explicit_regression_overrides_auto_detect(self):
        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        # Even a binary-looking CSV should honour the explicit override
        csv_path = _binary_csv()
        detected = DatasetLoader().detect_task_type(csv_path, "label", task_type=TaskType.regression)
        assert detected == TaskType.regression

    def test_explicit_classification_overrides_auto_detect(self):
        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        csv_path = _regression_csv()
        detected = DatasetLoader().detect_task_type(csv_path, "target", task_type=TaskType.classification)
        assert detected == TaskType.classification

    def test_load_still_returns_two_tuple(self):
        from agent.data_loader import DatasetLoader
        result = DatasetLoader().load(SYNTHETIC_CSV, "churn")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# EvaluationResult backward compatibility
# ---------------------------------------------------------------------------

class TestEvaluationResultBackwardCompat:
    def test_old_auc_kwarg_accepted(self):
        from tools.schemas import EvaluationResult
        result = EvaluationResult(auc=0.75, f1=0.60, shap_values={"x": 0.1}, feature_names=["x"])
        assert result.primary_metric == 0.75

    def test_old_f1_kwarg_accepted(self):
        from tools.schemas import EvaluationResult
        result = EvaluationResult(auc=0.75, f1=0.60, shap_values={"x": 0.1}, feature_names=["x"])
        assert result.secondary_metric == 0.60

    def test_auc_property_alias(self):
        from tools.schemas import EvaluationResult
        result = EvaluationResult(primary_metric=0.80, secondary_metric=0.65, shap_values={}, feature_names=[])
        assert result.auc == 0.80

    def test_f1_property_alias(self):
        from tools.schemas import EvaluationResult
        result = EvaluationResult(primary_metric=0.80, secondary_metric=0.65, shap_values={}, feature_names=[])
        assert result.f1 == 0.65


# ---------------------------------------------------------------------------
# AgentTrace backward compatibility
# ---------------------------------------------------------------------------

class TestAgentTraceBackwardCompat:
    def test_old_baseline_auc_kwarg_accepted(self):
        from tools.schemas import AgentTrace
        trace = AgentTrace(baseline_auc=0.70, final_auc=0.75, iterations=[], final_feature_set=[])
        assert trace.baseline_metric == 0.70

    def test_old_final_auc_kwarg_accepted(self):
        from tools.schemas import AgentTrace
        trace = AgentTrace(baseline_auc=0.70, final_auc=0.75, iterations=[], final_feature_set=[])
        assert trace.final_metric == 0.75

    def test_baseline_auc_property_alias(self):
        from tools.schemas import AgentTrace
        trace = AgentTrace(baseline_metric=0.70, final_metric=0.75, iterations=[], final_feature_set=[])
        assert trace.baseline_auc == 0.70

    def test_final_auc_property_alias(self):
        from tools.schemas import AgentTrace
        trace = AgentTrace(baseline_metric=0.70, final_metric=0.75, iterations=[], final_feature_set=[])
        assert trace.final_auc == 0.75

    def test_task_type_defaults_to_classification(self):
        from tools.schemas import AgentTrace, TaskType
        trace = AgentTrace(baseline_metric=0.70, final_metric=0.75, iterations=[], final_feature_set=[])
        assert trace.task_type == TaskType.classification

    def test_task_type_regression_stored(self):
        from tools.schemas import AgentTrace, TaskType
        trace = AgentTrace(
            baseline_metric=5.0,
            final_metric=4.5,
            iterations=[],
            final_feature_set=[],
            task_type=TaskType.regression,
        )
        assert trace.task_type == TaskType.regression


# ---------------------------------------------------------------------------
# EvaluateTool — regression path
# ---------------------------------------------------------------------------

class TestEvaluateToolRegression:
    def _reg_df(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "target": rng.standard_normal(n) * 5 + 20,
        })

    def test_returns_evaluation_result(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import EvaluationResult, TaskType
        result = EvaluateTool().evaluate(self._reg_df(), "target", TaskType.regression)
        assert isinstance(result, EvaluationResult)

    def test_task_type_is_regression(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import TaskType
        result = EvaluateTool().evaluate(self._reg_df(), "target", TaskType.regression)
        assert result.task_type == TaskType.regression

    def test_primary_metric_is_positive_rmse(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import TaskType
        result = EvaluateTool().evaluate(self._reg_df(), "target", TaskType.regression)
        # RMSE must be a positive float
        assert result.primary_metric > 0.0

    def test_secondary_metric_is_r2_in_range(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import TaskType
        result = EvaluateTool().evaluate(self._reg_df(), "target", TaskType.regression)
        # R² can be negative for bad models but is typically bounded above by 1
        assert result.secondary_metric <= 1.0

    def test_shap_values_populated(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import TaskType
        result = EvaluateTool().evaluate(self._reg_df(), "target", TaskType.regression)
        assert len(result.shap_values) == 2  # feat_a and feat_b

    def test_deterministic_across_calls(self):
        from tools.evaluate import EvaluateTool
        from tools.schemas import TaskType
        df = self._reg_df()
        r1 = EvaluateTool().evaluate(df, "target", TaskType.regression)
        r2 = EvaluateTool().evaluate(df, "target", TaskType.regression)
        assert r1.primary_metric == r2.primary_metric


# ---------------------------------------------------------------------------
# LeakageDetector — regression MI branch
# ---------------------------------------------------------------------------

class TestLeakageDetectorRegression:
    def test_non_leaking_feature_passes(self):
        from agent.leakage_detector import LeakageDetector
        from tools.schemas import TaskType
        rng = np.random.default_rng(0)
        n = 300
        feat = pd.Series(rng.standard_normal(n))
        target = pd.Series(rng.standard_normal(n))
        result = LeakageDetector().is_leaking(feat, target, "my_feat", "target", TaskType.regression)
        assert result.is_leaking is False

    def test_name_check_fires_for_regression(self):
        from agent.leakage_detector import LeakageDetector
        from tools.schemas import TaskType
        rng = np.random.default_rng(0)
        n = 300
        feat = pd.Series(rng.standard_normal(n))
        target = pd.Series(rng.standard_normal(n))
        result = LeakageDetector().is_leaking(feat, target, "target_squared", "target", TaskType.regression)
        assert result.is_leaking is True

    def test_high_correlation_leaks_for_regression(self):
        from agent.leakage_detector import LeakageDetector
        from tools.schemas import TaskType
        rng = np.random.default_rng(0)
        n = 300
        target = pd.Series(rng.standard_normal(n))
        feat = target * 1.001  # near-perfect correlation
        result = LeakageDetector().is_leaking(feat, target, "near_perfect", "y", TaskType.regression)
        assert result.is_leaking is True

    def test_high_mi_regression_triggers_leak(self):
        """mutual_info_regression > 0.9 must flag is_leaking=True."""
        from agent.leakage_detector import LeakageDetector
        from tools.schemas import TaskType
        rng = np.random.default_rng(0)
        n = 300
        feat = pd.Series(rng.standard_normal(n))
        target = pd.Series(rng.standard_normal(n))
        with patch("agent.leakage_detector.mutual_info_regression", return_value=np.array([0.95])):
            result = LeakageDetector().is_leaking(feat, target, "engineered_feat", "price", TaskType.regression)
        assert result.is_leaking is True
        assert "Mutual information" in result.reason


# ---------------------------------------------------------------------------
# detect_task_type — boundary condition
# ---------------------------------------------------------------------------

class TestDetectTaskTypeBoundary:
    def test_exactly_20_unique_values_is_classification(self, tmp_path):
        """The rule is > 20 unique values → regression; exactly 20 must remain classification."""
        n = 200
        # Cycle through exactly 20 distinct integer values
        target = [i % 20 for i in range(n)]
        df = pd.DataFrame({"feat_a": np.random.default_rng(0).standard_normal(n), "target": target})
        csv_path = str(tmp_path / "data.csv")
        df.to_csv(csv_path, index=False)

        from agent.data_loader import DatasetLoader
        from tools.schemas import TaskType
        detected = DatasetLoader().detect_task_type(csv_path, "target")
        assert detected == TaskType.classification


# ---------------------------------------------------------------------------
# LLMReasoner — task_type note injected into system prompt
# ---------------------------------------------------------------------------

class TestLLMReasonerTaskTypeNote:
    def _make_profile(self):
        from tools.schemas import DatasetProfile
        return DatasetProfile(
            row_count=100,
            column_count=3,
            target_col="target",
            feature_cols=["a", "b"],
            missing_rate={"a": 0.0, "b": 0.0},
            dtypes={"a": "float64", "b": "float64", "target": "float64"},
        )

    def _make_shap(self):
        from tools.schemas import ShapSummary
        return ShapSummary(ranked_features=[], top_3_summary="No prior data.")

    def _valid_llm_response(self):
        import json
        return json.dumps({
            "hypothesis": "test hypothesis",
            "feature_name": "new_feat",
            "transformation_code": "df['new_feat'] = df['a'] + 1",
            "decision_rationale": "test rationale",
        })

    def test_regression_note_in_system_arg(self):
        """reason() must append 'regression' to the system prompt for TaskType.regression."""
        from agent.llm_reasoner import LLMReasoner
        from tools.schemas import TaskType

        captured = {}

        def fake_complete(system, user):
            captured["system"] = system
            return self._valid_llm_response()

        reasoner = LLMReasoner()
        with patch.object(reasoner._client, "complete", side_effect=fake_complete):
            reasoner.reason(
                self._make_profile(),
                self._make_shap(),
                [],
                [],
                task_type=TaskType.regression,
            )

        assert "regression" in captured["system"]

    def test_classification_note_in_system_arg(self):
        """reason() must include 'classification' in the system prompt for TaskType.classification."""
        from agent.llm_reasoner import LLMReasoner
        from tools.schemas import TaskType

        captured = {}

        def fake_complete(system, user):
            captured["system"] = system
            return self._valid_llm_response()

        reasoner = LLMReasoner()
        with patch.object(reasoner._client, "complete", side_effect=fake_complete):
            reasoner.reason(
                self._make_profile(),
                self._make_shap(),
                [],
                [],
                task_type=TaskType.classification,
            )

        assert "classification" in captured["system"]


# ---------------------------------------------------------------------------
# AgentLoop — regression keep / discard decision
# ---------------------------------------------------------------------------

def _make_reg_loop_df(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "target": rng.standard_normal(n) * 5 + 20,
    })


def _make_reg_eval(rmse: float) -> "EvaluationResult":
    from tools.schemas import EvaluationResult, TaskType
    return EvaluationResult(
        primary_metric=rmse,
        secondary_metric=0.30,
        shap_values={"feat_a": 0.10, "feat_b": 0.07},
        feature_names=["feat_a", "feat_b"],
        task_type=TaskType.regression,
    )


def _run_regression_loop(tmp_path, baseline_rmse: float, iter_rmse: float) -> "AgentTrace":
    """Run AgentLoop for 1 iteration with mocked tools; returns AgentTrace."""
    from agent.loop import AgentLoop
    from tools.schemas import (
        ExecuteResult, FeatureShapEntry, LeakageResult, ReasoningOutput, ShapSummary, TaskType,
    )

    base_df = _make_reg_loop_df()
    exec_df = base_df.copy()
    exec_df["new_feat"] = 0.0

    loader_mock = MagicMock()
    loader_mock.load.return_value = (base_df.copy(), base_df.copy())
    loader_mock.detect_task_type.return_value = TaskType.regression

    eval_mock = MagicMock()
    eval_mock.evaluate.side_effect = [
        _make_reg_eval(baseline_rmse),
        _make_reg_eval(iter_rmse),
    ]

    shap_mock = MagicMock()
    shap_mock.format_for_llm.return_value = ShapSummary(
        ranked_features=[FeatureShapEntry(feature_name="feat_a", mean_abs_shap=0.1, rank=1)],
        top_3_summary="feat_a (shap=0.100)",
    )

    reasoner_mock = MagicMock()
    reasoner_mock.reason.return_value = ReasoningOutput(
        hypothesis="test",
        feature_name="new_feat",
        transformation_code="df['new_feat'] = 0.0",
        decision_rationale="test",
    )

    execute_mock = MagicMock()
    execute_mock.execute.return_value = ExecuteResult(
        success=True, new_columns=["new_feat"], error_message=None, output_df=exec_df,
    )

    leak_mock = MagicMock()
    leak_mock.is_leaking.return_value = LeakageResult(is_leaking=False, reason=None)

    with patch("agent.loop.DatasetLoader", MagicMock(return_value=loader_mock)), \
         patch("agent.loop.EvaluateTool", MagicMock(return_value=eval_mock)), \
         patch("agent.loop.ShapTool", MagicMock(return_value=shap_mock)), \
         patch("agent.loop.ProfileTool", MagicMock(return_value=MagicMock())), \
         patch("agent.loop.LLMReasoner", MagicMock(return_value=reasoner_mock)), \
         patch("agent.loop.ExecuteTool", MagicMock(return_value=execute_mock)), \
         patch("agent.loop.LeakageDetector", MagicMock(return_value=leak_mock)), \
         patch("agent.loop.OUTPUTS_DIR", tmp_path / "outputs"):
        return AgentLoop().run("fake.csv", "target", max_iter=1, task_type="regression")


class TestAgentLoopRegressionDecision:
    def test_rmse_decrease_labelled_kept(self, tmp_path):
        """Lower RMSE (negative delta) must produce decision='kept'."""
        trace = _run_regression_loop(tmp_path, baseline_rmse=5.0, iter_rmse=4.5)
        assert trace.iterations[0].decision == "kept"

    def test_rmse_increase_labelled_discarded(self, tmp_path):
        """Higher RMSE (positive delta) must produce decision='discarded'."""
        trace = _run_regression_loop(tmp_path, baseline_rmse=5.0, iter_rmse=5.5)
        assert trace.iterations[0].decision == "discarded"

    def test_trace_task_type_is_regression(self, tmp_path):
        """AgentTrace returned from a regression run must carry task_type=regression."""
        from tools.schemas import TaskType
        trace = _run_regression_loop(tmp_path, baseline_rmse=5.0, iter_rmse=4.5)
        assert trace.task_type == TaskType.regression

    def test_auc_delta_reflects_rmse_delta(self, tmp_path):
        """IterationRecord.auc_delta stores primary-metric delta (RMSE after − before)."""
        trace = _run_regression_loop(tmp_path, baseline_rmse=5.0, iter_rmse=4.5)
        assert abs(trace.iterations[0].auc_delta - (-0.5)) < 1e-9
