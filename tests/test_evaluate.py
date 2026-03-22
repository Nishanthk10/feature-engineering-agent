import numpy as np
import pandas as pd
import pytest

from tools.evaluate import EvaluateTool
from tools.schemas import EvaluationResult


def make_df(n_rows: int = 200, n_features: int = 4, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = {f"feat_{i}": rng.standard_normal(n_rows) for i in range(n_features)}
    # Make target separable so AUC > 0.5 is reliably achieved
    signal = sum(data[f"feat_{i}"] for i in range(n_features))
    data["target"] = (signal > 0).astype(int)
    return pd.DataFrame(data)


class TestEvaluateToolResult:
    def test_returns_evaluation_result_instance(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert isinstance(result, EvaluationResult)

    def test_all_fields_populated(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert isinstance(result.auc, float)
        assert isinstance(result.f1, float)
        assert isinstance(result.shap_values, dict)
        assert isinstance(result.feature_names, list)
        assert len(result.feature_names) > 0
        assert len(result.shap_values) > 0

    def test_auc_between_0_5_and_1(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert 0.5 <= result.auc <= 1.0

    def test_shap_values_has_one_key_per_feature(self):
        df = make_df(n_features=4)
        result = EvaluateTool().evaluate(df, "target")
        expected_features = [c for c in df.columns if c != "target"]
        assert set(result.shap_values.keys()) == set(expected_features)
        assert result.feature_names == expected_features

    def test_f1_between_0_and_1(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert isinstance(result.f1, float)
        assert 0.0 <= result.f1 <= 1.0

    def test_target_col_absent_from_feature_names(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert "target" not in result.feature_names

    def test_all_shap_values_non_negative(self):
        df = make_df()
        result = EvaluateTool().evaluate(df, "target")
        assert all(v >= 0.0 for v in result.shap_values.values())

    def test_single_feature_returns_one_shap_key(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "feat_0": rng.standard_normal(200),
            "target": rng.integers(0, 2, 200),
        })
        result = EvaluateTool().evaluate(df, "target")
        assert list(result.shap_values.keys()) == ["feat_0"]

    def test_determinism_same_auc_on_repeated_calls(self):
        df = make_df()
        result_a = EvaluateTool().evaluate(df, "target")
        result_b = EvaluateTool().evaluate(df, "target")
        assert result_a.auc == result_b.auc

    def test_determinism_same_shap_values_on_repeated_calls(self):
        df = make_df()
        result_a = EvaluateTool().evaluate(df, "target")
        result_b = EvaluateTool().evaluate(df, "target")
        assert result_a.shap_values == result_b.shap_values
