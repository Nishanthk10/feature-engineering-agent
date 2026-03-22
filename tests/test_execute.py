import numpy as np
import pandas as pd
import pytest

from tools.execute import ExecuteTool


def make_df(n_rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n_rows),
        "feat_b": rng.standard_normal(n_rows),
    })


class TestExecuteToolHappyPath:
    def test_success_true_for_valid_code(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['feat_sum'] = df['feat_a'] + df['feat_b']")
        assert result.success is True

    def test_new_column_in_new_columns(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['feat_sum'] = df['feat_a'] + df['feat_b']")
        assert "feat_sum" in result.new_columns

    def test_output_df_contains_new_column(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['feat_ratio'] = df['feat_a'] / (df['feat_b'] + 1e-9)")
        assert result.output_df is not None
        assert "feat_ratio" in result.output_df.columns

    def test_error_message_is_none_on_success(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['x'] = 1")
        assert result.error_message is None

    def test_original_df_not_modified(self):
        df = make_df()
        original_cols = list(df.columns)
        ExecuteTool().execute(df, "df['new_col'] = df['feat_a'] * 2")
        assert list(df.columns) == original_cols

    def test_no_new_columns_returns_empty_list(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['feat_a'] = df['feat_a'] * 2")
        assert result.success is True
        assert result.new_columns == []

    def test_multiple_new_columns_all_returned(self):
        df = make_df()
        code = (
            "df['col_x'] = df['feat_a'] + df['feat_b']\n"
            "df['col_y'] = df['feat_a'] - df['feat_b']\n"
            "df['col_z'] = df['feat_a'] * df['feat_b']"
        )
        result = ExecuteTool().execute(df, code)
        assert result.success is True
        assert set(result.new_columns) == {"col_x", "col_y", "col_z"}

    def test_output_df_row_count_matches_input(self):
        df = make_df(n_rows=150)
        result = ExecuteTool().execute(df, "df['new_col'] = df['feat_a'] * 2")
        assert result.output_df is not None
        assert len(result.output_df) == len(df)

    def test_output_df_preserves_original_columns(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['new_col'] = df['feat_a'] + 1")
        assert result.output_df is not None
        assert "feat_a" in result.output_df.columns
        assert "feat_b" in result.output_df.columns


class TestExecuteToolErrors:
    def test_invalid_syntax_returns_success_false(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['x'] = )(invalid syntax")
        assert result.success is False

    def test_invalid_syntax_has_error_message(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['x'] = )(invalid syntax")
        assert result.error_message is not None
        assert len(result.error_message) > 0

    def test_invalid_syntax_output_df_is_none(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['x'] = )(invalid syntax")
        assert result.output_df is None

    def test_disallowed_import_returns_success_false(self):
        df = make_df()
        result = ExecuteTool().execute(df, "import os\ndf['x'] = 1")
        assert result.success is False

    def test_disallowed_import_has_error_message(self):
        df = make_df()
        result = ExecuteTool().execute(df, "import os\ndf['x'] = 1")
        assert result.error_message is not None

    def test_disallowed_import_subprocess_returns_error(self):
        df = make_df()
        result = ExecuteTool().execute(df, "import subprocess\ndf['x'] = 1")
        assert result.success is False

    def test_disallowed_from_import_form_blocked(self):
        df = make_df()
        result = ExecuteTool().execute(df, "from os import path\ndf['x'] = 1")
        assert result.success is False

    def test_runtime_error_returns_success_false(self):
        df = make_df()
        result = ExecuteTool().execute(df, "df['x'] = df['nonexistent_col'] + 1")
        assert result.success is False


class TestExecuteToolTimeout:
    @pytest.mark.slow
    def test_timeout_returns_success_false(self):
        df = make_df()
        code = "while True:\n    x = 1 + 1"
        result = ExecuteTool().execute(df, code)
        assert result.success is False

    @pytest.mark.slow
    def test_timeout_has_error_message(self):
        df = make_df()
        code = "while True:\n    x = 1 + 1"
        result = ExecuteTool().execute(df, code)
        assert result.error_message is not None
