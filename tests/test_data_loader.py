import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from agent.data_loader import DatasetLoader


def make_csv(tmp_path: pathlib.Path, rows: int = 120, target: str = "label") -> pathlib.Path:
    """Write a minimal valid CSV with `rows` rows and a target column."""
    lines = [f"feature_a,feature_b,{target}"]
    for i in range(rows):
        lines.append(f"{i},{i * 2},{i % 2}")
    p = tmp_path / "dataset.csv"
    p.write_text("\n".join(lines))
    return p


class TestDatasetLoaderHappyPath:
    def test_returns_two_dataframes(self, tmp_path):
        csv = make_csv(tmp_path)
        loader = DatasetLoader()
        original, working = loader.load(str(csv), "label")
        assert isinstance(original, pd.DataFrame)
        assert isinstance(working, pd.DataFrame)

    def test_both_contain_target_column(self, tmp_path):
        csv = make_csv(tmp_path)
        original, working = DatasetLoader().load(str(csv), "label")
        assert "label" in original.columns
        assert "label" in working.columns

    def test_row_count_matches_source(self, tmp_path):
        csv = make_csv(tmp_path, rows=120)
        original, working = DatasetLoader().load(str(csv), "label")
        assert len(original) == 120
        assert len(working) == 120


class TestDatasetLoaderErrors:
    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            DatasetLoader().load(str(tmp_path / "nonexistent.csv"), "label")

    def test_missing_target_column_raises_value_error(self, tmp_path):
        csv = make_csv(tmp_path)
        with pytest.raises(ValueError, match="Target column"):
            DatasetLoader().load(str(csv), "no_such_column")

    def test_too_few_rows_raises_value_error(self, tmp_path):
        csv = make_csv(tmp_path, rows=50)
        with pytest.raises(ValueError, match="fewer than"):
            DatasetLoader().load(str(csv), "label")

    def test_exactly_99_rows_raises_value_error(self, tmp_path):
        csv = make_csv(tmp_path, rows=99)
        with pytest.raises(ValueError, match="fewer than"):
            DatasetLoader().load(str(csv), "label")

    def test_exactly_100_rows_is_accepted(self, tmp_path):
        csv = make_csv(tmp_path, rows=100)
        original, working = DatasetLoader().load(str(csv), "label")
        assert len(original) == 100

    def test_file_over_500mb_raises_value_error(self, tmp_path):
        csv = make_csv(tmp_path)
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 500 * 1024 * 1024 + 1
            with pytest.raises(ValueError, match="500 MB"):
                DatasetLoader().load(str(csv), "label")


class TestDatasetLoaderIndependence:
    def test_working_df_mutation_does_not_affect_original(self, tmp_path):
        csv = make_csv(tmp_path)
        original, working = DatasetLoader().load(str(csv), "label")
        original_value = original.at[0, "feature_a"]

        working.at[0, "feature_a"] = -9999

        assert original.at[0, "feature_a"] == original_value

    def test_original_and_working_are_different_objects(self, tmp_path):
        csv = make_csv(tmp_path)
        original, working = DatasetLoader().load(str(csv), "label")
        assert original is not working

    def test_original_df_mutation_does_not_affect_working(self, tmp_path):
        csv = make_csv(tmp_path)
        original, working = DatasetLoader().load(str(csv), "label")
        working_value = working.at[0, "feature_b"]

        original.at[0, "feature_b"] = -9999

        assert working.at[0, "feature_b"] == working_value
