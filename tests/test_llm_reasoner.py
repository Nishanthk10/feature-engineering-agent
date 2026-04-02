import json
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

import pytest

from agent.llm_reasoner import LLMClient, LLMReasoner, SYSTEM_PROMPT
from tools.schemas import (
    DatasetProfile,
    EvaluationResult,
    FeatureShapEntry,
    IterationRecord,
    ReasoningOutput,
    ShapSummary,
)


VALID_RESPONSE = {
    "hypothesis": "Ratio of income to balance may capture financial stress.",
    "feature_name": "income_balance_ratio",
    "transformation_code": "df['income_balance_ratio'] = df['income'] / (df['balance'] + 1e-9)",
    "decision_rationale": "SHAP shows income and balance are top features; their ratio adds signal.",
}


def make_profile() -> DatasetProfile:
    return DatasetProfile(
        row_count=200,
        column_count=4,
        target_col="target",
        feature_cols=["age", "income", "balance"],
        missing_rate={"age": 0.0, "income": 0.0, "balance": 0.0},
        dtypes={"age": "int64", "income": "int64", "balance": "float64", "target": "int64"},
    )


def make_shap_summary() -> ShapSummary:
    return ShapSummary(
        ranked_features=[
            FeatureShapEntry(feature_name="income", mean_abs_shap=0.12, rank=1),
            FeatureShapEntry(feature_name="balance", mean_abs_shap=0.08, rank=2),
            FeatureShapEntry(feature_name="age", mean_abs_shap=0.05, rank=3),
        ],
        top_3_summary="Top features by importance: (1) income (shap=0.120), (2) balance (shap=0.080), (3) age (shap=0.050)",
    )


def make_history() -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=1,
            hypothesis="Age squared",
            feature_name="balance_ratio",
            transformation_code="df['balance_ratio'] = df['balance'] / df['income']",
            auc_before=0.71,
            auc_after=0.74,
            auc_delta=0.003,
            shap_summary=ShapSummary(
                ranked_features=[FeatureShapEntry(feature_name="balance_ratio", mean_abs_shap=0.09, rank=1)],
                top_3_summary="Top features by importance: (1) balance_ratio (shap=0.090)",
            ),
            decision="kept",
            error_message=None,
            status="completed",
        )
    ]


def make_record(iteration: int) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        hypothesis=f"Hypothesis {iteration}",
        feature_name=f"feat_{iteration}",
        transformation_code=f"df['feat_{iteration}'] = df['balance'] * {iteration}",
        auc_before=0.70,
        auc_after=0.70 + iteration * 0.001,
        auc_delta=iteration * 0.001,
        shap_summary=ShapSummary(
            ranked_features=[FeatureShapEntry(feature_name=f"feat_{iteration}", mean_abs_shap=0.05, rank=1)],
            top_3_summary=f"Top features by importance: (1) feat_{iteration} (shap=0.050)",
        ),
        decision="kept",
        error_message=None,
        status="completed",
    )


def _patch_complete(response_text: str):
    """Patch LLMClient.complete to return response_text."""
    return patch.object(LLMClient, "complete", return_value=response_text)


class TestLLMReasonerValidResponse:
    def test_valid_json_parses_into_reasoning_output(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), make_history(), ["age_sq"], iteration_number=1)
        assert isinstance(result, ReasoningOutput)

    def test_hypothesis_field_populated(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), make_history(), [], iteration_number=1)
        assert result.hypothesis == VALID_RESPONSE["hypothesis"]

    def test_feature_name_field_populated(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), make_history(), [], iteration_number=1)
        assert result.feature_name == VALID_RESPONSE["feature_name"]

    def test_transformation_code_present_and_non_empty(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), make_history(), [], iteration_number=1)
        assert isinstance(result.transformation_code, str)
        assert len(result.transformation_code.strip()) > 0

    def test_decision_rationale_field_populated(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), make_history(), [], iteration_number=1)
        assert result.decision_rationale == VALID_RESPONSE["decision_rationale"]


class TestLLMReasonerInvalidResponse:
    def test_malformed_json_raises_value_error(self):
        with _patch_complete("not valid json {"):
            with pytest.raises(ValueError):
                LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)

    def test_value_error_includes_raw_response(self):
        raw = "definitely not json"
        with _patch_complete(raw):
            with pytest.raises(ValueError, match=raw):
                LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)

    def test_empty_response_raises_value_error(self):
        with _patch_complete(""):
            with pytest.raises(ValueError):
                LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)

    def test_missing_required_key_raises_validation_error(self):
        incomplete = {k: v for k, v in VALID_RESPONSE.items() if k != "transformation_code"}
        with _patch_complete(json.dumps(incomplete)):
            with pytest.raises(ValidationError):
                LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)


class TestLLMReasonerAPICall:
    def test_complete_called_once_per_reason_call(self):
        with patch.object(LLMClient, "complete", return_value=json.dumps(VALID_RESPONSE)) as mock_complete:
            LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)
        mock_complete.assert_called_once()

    def test_history_of_5_truncates_to_last_3(self):
        history = [make_record(i) for i in range(1, 6)]  # iterations 1–5
        with patch.object(LLMClient, "complete", return_value=json.dumps(VALID_RESPONSE)) as mock_complete:
            LLMReasoner().reason(make_profile(), make_shap_summary(), history, [], iteration_number=1)
        _, user_prompt = mock_complete.call_args[0]
        assert "Hypothesis 3" in user_prompt
        assert "Hypothesis 4" in user_prompt
        assert "Hypothesis 5" in user_prompt
        assert "Hypothesis 1" not in user_prompt
        assert "Hypothesis 2" not in user_prompt

    def test_empty_history_does_not_raise(self):
        with _patch_complete(json.dumps(VALID_RESPONSE)):
            result = LLMReasoner().reason(make_profile(), make_shap_summary(), [], [], iteration_number=1)
        assert isinstance(result, ReasoningOutput)
