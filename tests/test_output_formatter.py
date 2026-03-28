import csv
import pathlib

import pytest
from pydantic import ValidationError

from agent.output_formatter import OutputFormatter
from tools.schemas import (
    AgentTrace,
    DatasetProfile,
    FeatureCandidate,
    FeatureShapEntry,
    IterationRecord,
    ShapSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(target_col: str = "target") -> DatasetProfile:
    return DatasetProfile(
        row_count=200,
        column_count=4,
        target_col=target_col,
        feature_cols=["age", "income", "balance"],
        missing_rate={"age": 0.0, "income": 0.0, "balance": 0.0},
        dtypes={"age": "int64", "income": "float64", "balance": "float64", "target": "int64"},
    )


def _make_shap_summary(feature_name: str, shap_val: float = 0.12) -> ShapSummary:
    return ShapSummary(
        ranked_features=[FeatureShapEntry(feature_name=feature_name, mean_abs_shap=shap_val, rank=1)],
        top_3_summary=f"Top features by importance: (1) {feature_name} (shap={shap_val:.3f})",
    )


def _make_record(
    iteration: int,
    feature_name: str,
    decision: str,
    auc_delta: float = 0.01,
    shap_val: float = 0.12,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        hypothesis=f"Hypothesis for {feature_name}",
        feature_name=feature_name,
        transformation_code=f"df['{feature_name}'] = df['age'] * {iteration}",
        auc_before=0.70,
        auc_after=0.70 + auc_delta,
        auc_delta=auc_delta,
        shap_summary=_make_shap_summary(feature_name, shap_val),
        decision=decision,
        error_message=None,
        status="completed",
    )


def _make_trace(
    iterations: list[IterationRecord],
    baseline_auc: float = 0.70,
    final_auc: float = 0.75,
) -> AgentTrace:
    return AgentTrace(
        baseline_auc=baseline_auc,
        iterations=iterations,
        final_feature_set=[r.feature_name for r in iterations if r.decision == "kept"],
        final_auc=final_auc,
    )


def _formatter() -> OutputFormatter:
    return OutputFormatter()


# ---------------------------------------------------------------------------
# FeatureCandidate validation
# ---------------------------------------------------------------------------

class TestFeatureCandidateValidation:
    def test_empty_hypothesis_raises(self):
        with pytest.raises(ValidationError):
            FeatureCandidate(
                name="feat",
                transformation_code="df['feat'] = 1",
                hypothesis="",
                mean_abs_shap=0.1,
                auc_delta=0.01,
                decision="kept",
            )

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            FeatureCandidate(
                name="",
                transformation_code="df['feat'] = 1",
                hypothesis="Some hypothesis",
                mean_abs_shap=0.1,
                auc_delta=0.01,
                decision="kept",
            )

    def test_empty_transformation_code_raises(self):
        with pytest.raises(ValidationError):
            FeatureCandidate(
                name="feat",
                transformation_code="",
                hypothesis="Some hypothesis",
                mean_abs_shap=0.1,
                auc_delta=0.01,
                decision="kept",
            )

    def test_whitespace_only_hypothesis_raises(self):
        with pytest.raises(ValidationError):
            FeatureCandidate(
                name="feat",
                transformation_code="df['feat'] = 1",
                hypothesis="   ",
                mean_abs_shap=0.1,
                auc_delta=0.01,
                decision="kept",
            )

    def test_valid_candidate_does_not_raise(self):
        fc = FeatureCandidate(
            name="income_ratio",
            transformation_code="df['income_ratio'] = df['income'] / df['balance']",
            hypothesis="Ratio captures financial stress.",
            mean_abs_shap=0.09,
            auc_delta=0.005,
            decision="kept",
        )
        assert fc.name == "income_ratio"


# ---------------------------------------------------------------------------
# FormattedOutput content
# ---------------------------------------------------------------------------

class TestFormattedOutputKeptFeatures:
    def test_kept_features_only_contains_kept_decisions(self):
        iterations = [
            _make_record(1, "feat_kept", "kept"),
            _make_record(2, "feat_discarded", "discarded"),
            _make_record(3, "feat_error", "error"),
        ]
        trace = _make_trace(iterations, final_auc=0.71)
        result = _formatter().format(trace, _make_profile())
        assert len(result.kept_features) == 1
        assert result.kept_features[0].name == "feat_kept"

    def test_kept_features_all_have_decision_kept(self):
        iterations = [
            _make_record(1, "feat_a", "kept"),
            _make_record(2, "feat_b", "kept"),
            _make_record(3, "feat_c", "discarded"),
        ]
        trace = _make_trace(iterations, final_auc=0.72)
        result = _formatter().format(trace, _make_profile())
        assert all(f.decision == "kept" for f in result.kept_features)

    def test_no_kept_features_returns_empty_list(self):
        iterations = [
            _make_record(1, "feat_a", "discarded"),
            _make_record(2, "feat_b", "error"),
        ]
        trace = _make_trace(iterations, final_auc=0.70)
        result = _formatter().format(trace, _make_profile())
        assert result.kept_features == []


class TestFormattedOutputMetrics:
    def test_auc_lift_equals_final_minus_baseline(self):
        trace = _make_trace([], baseline_auc=0.68, final_auc=0.74)
        result = _formatter().format(trace, _make_profile())
        assert result.auc_lift == pytest.approx(0.74 - 0.68)

    def test_baseline_auc_propagated(self):
        trace = _make_trace([], baseline_auc=0.65, final_auc=0.70)
        result = _formatter().format(trace, _make_profile())
        assert result.baseline_auc == pytest.approx(0.65)

    def test_final_auc_propagated(self):
        trace = _make_trace([], baseline_auc=0.65, final_auc=0.70)
        result = _formatter().format(trace, _make_profile())
        assert result.final_auc == pytest.approx(0.70)


class TestReportText:
    def test_report_text_contains_all_kept_feature_names(self):
        iterations = [
            _make_record(1, "income_ratio", "kept"),
            _make_record(2, "age_sq", "kept"),
            _make_record(3, "balance_log", "discarded"),
        ]
        trace = _make_trace(iterations, final_auc=0.75)
        result = _formatter().format(trace, _make_profile())
        assert "income_ratio" in result.report_text
        assert "age_sq" in result.report_text
        assert "balance_log" not in result.report_text

    def test_report_text_contains_baseline_auc(self):
        trace = _make_trace([], baseline_auc=0.6800, final_auc=0.7100)
        result = _formatter().format(trace, _make_profile())
        assert "0.6800" in result.report_text

    def test_report_text_contains_final_auc(self):
        trace = _make_trace([], baseline_auc=0.68, final_auc=0.7100)
        result = _formatter().format(trace, _make_profile())
        assert "0.7100" in result.report_text

    def test_report_text_contains_iteration_count(self):
        iterations = [_make_record(i, f"feat_{i}", "kept") for i in range(1, 4)]
        trace = _make_trace(iterations, final_auc=0.73)
        result = _formatter().format(trace, _make_profile())
        assert "3" in result.report_text


# ---------------------------------------------------------------------------
# SHAP value resolution
# ---------------------------------------------------------------------------

class TestShapLookup:
    def test_shap_fallback_zero_when_feature_absent_from_ranked_list(self):
        """If feature_name is not in ranked_features, mean_abs_shap defaults to 0.0."""
        shap_summary = ShapSummary(
            ranked_features=[
                FeatureShapEntry(feature_name="other_feat", mean_abs_shap=0.15, rank=1),
            ],
            top_3_summary="Top features: (1) other_feat (shap=0.150)",
        )
        record = IterationRecord(
            iteration=1,
            hypothesis="Test hypothesis",
            feature_name="my_feat",           # not present in ranked_features
            transformation_code="df['my_feat'] = 1",
            auc_before=0.70,
            auc_after=0.71,
            auc_delta=0.01,
            shap_summary=shap_summary,
            decision="kept",
            error_message=None,
            status="completed",
        )
        trace = _make_trace([record], final_auc=0.71)
        result = _formatter().format(trace, _make_profile())
        assert result.kept_features[0].mean_abs_shap == pytest.approx(0.0)

    def test_shap_value_selected_by_feature_name_not_first_entry(self):
        """Formatter picks the entry matching feature_name, not always index 0."""
        shap_summary = ShapSummary(
            ranked_features=[
                FeatureShapEntry(feature_name="other_feat", mean_abs_shap=0.30, rank=1),
                FeatureShapEntry(feature_name="target_feat", mean_abs_shap=0.11, rank=2),
                FeatureShapEntry(feature_name="third_feat", mean_abs_shap=0.05, rank=3),
            ],
            top_3_summary="Top features: (1) other_feat, (2) target_feat, (3) third_feat",
        )
        record = IterationRecord(
            iteration=1,
            hypothesis="Test hypothesis",
            feature_name="target_feat",       # second entry, not first
            transformation_code="df['target_feat'] = 1",
            auc_before=0.70,
            auc_after=0.71,
            auc_delta=0.01,
            shap_summary=shap_summary,
            decision="kept",
            error_message=None,
            status="completed",
        )
        trace = _make_trace([record], final_auc=0.71)
        result = _formatter().format(trace, _make_profile())
        assert result.kept_features[0].mean_abs_shap == pytest.approx(0.11)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = ["name", "hypothesis", "mean_abs_shap", "auc_delta", "decision", "transformation_code"]


def _write_csv(formatted, path: pathlib.Path) -> None:
    """Replicate the CSV writing logic from run_agent.py for testing."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for feat in formatted.kept_features:
            writer.writerow({
                "name": feat.name,
                "hypothesis": feat.hypothesis,
                "mean_abs_shap": feat.mean_abs_shap,
                "auc_delta": feat.auc_delta,
                "decision": feat.decision,
                "transformation_code": feat.transformation_code,
            })


class TestCSVOutput:
    def test_csv_file_exists_after_write(self, tmp_path):
        trace = _make_trace([_make_record(1, "feat_a", "kept")], final_auc=0.71)
        formatted = _formatter().format(trace, _make_profile())
        csv_path = tmp_path / "final_features.csv"
        _write_csv(formatted, csv_path)
        assert csv_path.exists()

    def test_csv_has_correct_column_headers(self, tmp_path):
        trace = _make_trace([_make_record(1, "feat_a", "kept")], final_auc=0.71)
        formatted = _formatter().format(trace, _make_profile())
        csv_path = tmp_path / "final_features.csv"
        _write_csv(formatted, csv_path)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _CSV_FIELDNAMES

    def test_csv_has_one_row_per_kept_feature(self, tmp_path):
        iterations = [
            _make_record(1, "feat_a", "kept"),
            _make_record(2, "feat_b", "discarded"),
            _make_record(3, "feat_c", "kept"),
        ]
        trace = _make_trace(iterations, final_auc=0.73)
        formatted = _formatter().format(trace, _make_profile())
        csv_path = tmp_path / "final_features.csv"
        _write_csv(formatted, csv_path)
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # feat_a and feat_c only

    def test_csv_row_contains_correct_feature_name(self, tmp_path):
        trace = _make_trace([_make_record(1, "income_ratio", "kept")], final_auc=0.71)
        formatted = _formatter().format(trace, _make_profile())
        csv_path = tmp_path / "final_features.csv"
        _write_csv(formatted, csv_path)
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["name"] == "income_ratio"
