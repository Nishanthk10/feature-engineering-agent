"""
Tests for tools/mcp_server.py — all four MCP tools called directly,
bypassing MCP transport. No real MCP client needed.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from tools.mcp_server import (
    evaluate_features,
    execute_feature_code,
    get_shap_values,
    profile_dataset,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
SYNTHETIC_CSV = str(DATA_DIR / "synthetic_churn.csv")


def _small_df(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n),
        "target": rng.integers(0, 2, n),
    })


def _df_json(n: int = 150) -> str:
    return _small_df(n).to_json()


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

class TestImports:
    def test_profile_dataset_is_callable(self):
        assert callable(profile_dataset)

    def test_execute_feature_code_is_callable(self):
        assert callable(execute_feature_code)

    def test_evaluate_features_is_callable(self):
        assert callable(evaluate_features)

    def test_get_shap_values_is_callable(self):
        assert callable(get_shap_values)


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------

class TestProfileDataset:
    def test_returns_dict(self):
        result = profile_dataset(SYNTHETIC_CSV, "churn")
        assert isinstance(result, dict)

    def test_result_contains_row_count_key(self):
        result = profile_dataset(SYNTHETIC_CSV, "churn")
        assert "row_count" in result

    def test_row_count_matches_csv(self):
        result = profile_dataset(SYNTHETIC_CSV, "churn")
        assert result["row_count"] == 1000

    def test_result_contains_target_col(self):
        result = profile_dataset(SYNTHETIC_CSV, "churn")
        assert result["target_col"] == "churn"

    def test_target_absent_from_feature_cols(self):
        result = profile_dataset(SYNTHETIC_CSV, "churn")
        assert "churn" not in result["feature_cols"]


# ---------------------------------------------------------------------------
# execute_feature_code
# ---------------------------------------------------------------------------

class TestExecuteFeatureCode:
    def test_returns_dict(self):
        code = "df['new_feat'] = df['feat_a'] + df['feat_b']"
        result = execute_feature_code(_df_json(), code)
        assert isinstance(result, dict)

    def test_valid_code_returns_success_true(self):
        code = "df['new_feat'] = df['feat_a'] * 2"
        result = execute_feature_code(_df_json(), code)
        assert result["success"] is True

    def test_valid_code_new_columns_listed(self):
        code = "df['my_col'] = df['feat_a'] + 1"
        result = execute_feature_code(_df_json(), code)
        assert "my_col" in result["new_columns"]

    def test_invalid_code_returns_success_false(self):
        code = "raise ValueError('intentional error')"
        result = execute_feature_code(_df_json(), code)
        assert result["success"] is False

    def test_output_df_is_json_string_on_success(self):
        code = "df['new_feat'] = 1"
        result = execute_feature_code(_df_json(), code)
        assert isinstance(result["output_df"], str)
        recovered = pd.read_json(result["output_df"])
        assert "new_feat" in recovered.columns

    def test_output_df_is_none_on_failure(self):
        code = "raise RuntimeError('boom')"
        result = execute_feature_code(_df_json(), code)
        assert result["output_df"] is None


# ---------------------------------------------------------------------------
# evaluate_features
# ---------------------------------------------------------------------------

class TestEvaluateFeatures:
    def test_returns_dict(self):
        result = evaluate_features(_df_json(), "target")
        assert isinstance(result, dict)

    def test_result_contains_auc(self):
        result = evaluate_features(_df_json(), "target")
        assert "auc" in result

    def test_auc_is_float_between_0_and_1(self):
        result = evaluate_features(_df_json(), "target")
        assert 0.0 <= result["auc"] <= 1.0

    def test_result_contains_shap_values(self):
        result = evaluate_features(_df_json(), "target")
        assert "shap_values" in result
        assert isinstance(result["shap_values"], dict)


# ---------------------------------------------------------------------------
# get_shap_values
# ---------------------------------------------------------------------------

class TestGetShapValues:
    def _eval_result_json(self) -> str:
        eval_result = {
            "auc": 0.72,
            "f1": 0.65,
            "shap_values": {"feat_a": 0.10, "feat_b": 0.07},
            "feature_names": ["feat_a", "feat_b"],
        }
        return json.dumps(eval_result)

    def test_returns_dict(self):
        result = get_shap_values(self._eval_result_json())
        assert isinstance(result, dict)

    def test_result_contains_ranked_features(self):
        result = get_shap_values(self._eval_result_json())
        assert "ranked_features" in result

    def test_result_contains_top_3_summary(self):
        result = get_shap_values(self._eval_result_json())
        assert "top_3_summary" in result

    def test_ranked_features_sorted_by_shap_descending(self):
        result = get_shap_values(self._eval_result_json())
        shap_vals = [e["mean_abs_shap"] for e in result["ranked_features"]]
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_top_3_summary_contains_feature_name(self):
        result = get_shap_values(self._eval_result_json())
        assert "feat_a" in result["top_3_summary"]

    def test_malformed_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            get_shap_values("not valid json {")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            get_shap_values("")


# ---------------------------------------------------------------------------
# Security boundary
# ---------------------------------------------------------------------------

class TestSandboxSecurityAtMCPLayer:
    def test_blocked_import_returns_success_false(self):
        code = "import os; df['bad'] = os.getcwd()"
        result = execute_feature_code(_df_json(), code)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------

class TestMCPRegistration:
    def test_mcp_is_fastmcp_instance(self):
        from fastmcp import FastMCP
        from tools.mcp_server import mcp
        assert isinstance(mcp, FastMCP)

    def test_all_four_tools_are_registered(self):
        import asyncio
        from tools.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        registered = {t.name for t in tools}
        expected = {
            "profile_dataset",
            "execute_feature_code",
            "evaluate_features",
            "get_shap_values",
        }
        assert expected.issubset(registered)
