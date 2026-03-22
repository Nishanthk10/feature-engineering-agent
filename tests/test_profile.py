import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools.profile import ProfileTool
from tools.schemas import DatasetProfile


def make_df(n_rows: int = 150, n_features: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"feat_{i}": rng.standard_normal(n_rows) for i in range(n_features)}
    data["target"] = rng.integers(0, 2, n_rows)
    return pd.DataFrame(data)


class TestDatasetProfile:
    def test_correct_row_count(self):
        df = make_df(n_rows=150)
        profile = ProfileTool().profile(df, "target")
        assert profile.row_count == 150

    def test_correct_column_count(self):
        df = make_df(n_features=3)  # 3 features + 1 target = 4 columns
        profile = ProfileTool().profile(df, "target")
        assert profile.column_count == 4

    def test_target_col_not_in_feature_cols(self):
        df = make_df()
        profile = ProfileTool().profile(df, "target")
        assert "target" not in profile.feature_cols

    def test_missing_rate_keys_match_non_target_columns(self):
        df = make_df(n_features=3)
        profile = ProfileTool().profile(df, "target")
        expected = {c for c in df.columns if c != "target"}
        assert set(profile.missing_rate.keys()) == expected

    def test_returns_dataset_profile_instance(self):
        df = make_df()
        profile = ProfileTool().profile(df, "target")
        assert isinstance(profile, DatasetProfile)

    def test_missing_rate_reflects_actual_nulls(self):
        df = make_df(n_rows=200, n_features=2)
        df.loc[:19, "feat_0"] = np.nan  # 20/200 = 0.1
        profile = ProfileTool().profile(df, "target")
        assert abs(profile.missing_rate["feat_0"] - 0.1) < 1e-9

    def test_dtypes_contains_all_columns(self):
        df = make_df()
        profile = ProfileTool().profile(df, "target")
        assert set(profile.dtypes.keys()) == set(df.columns)

    def test_target_col_absent_from_missing_rate_keys(self):
        df = make_df()
        profile = ProfileTool().profile(df, "target")
        assert "target" not in profile.missing_rate

    def test_zero_missing_values_produces_zero_missing_rate(self):
        df = make_df()  # no NaNs
        profile = ProfileTool().profile(df, "target")
        assert all(v == 0.0 for v in profile.missing_rate.values())


def make_csv(path: pathlib.Path, n_rows: int = 200) -> pathlib.Path:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n_rows),
        "feat_b": rng.standard_normal(n_rows),
        "label": (rng.standard_normal(n_rows) > 0).astype(int),
    })
    path.write_text(df.to_csv(index=False))
    return path


ROOT = pathlib.Path(__file__).parent.parent


@pytest.mark.e2e
class TestRunAgent:
    def _run(self, *extra_args, outputs_dir: pathlib.Path = None, csv: pathlib.Path = None):
        cmd = [
            sys.executable, str(ROOT / "run_agent.py"),
            "--dataset", str(csv),
            "--target", "label",
            *extra_args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(outputs_dir.parent))

    def test_max_iter_argument_accepted(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label", "--max-iter", "3"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert result.returncode == 0

    def test_exits_with_code_0(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert result.returncode == 0

    def test_trace_json_exists_after_run(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert (tmp_path / "outputs" / "trace.json").exists()

    def test_trace_json_has_correct_structure(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        trace = json.loads((tmp_path / "outputs" / "trace.json").read_text())
        assert isinstance(trace, list) and len(trace) >= 1
        entry = trace[0]
        assert entry["iteration"] == 0
        assert entry["status"] == "baseline"
        assert isinstance(entry["auc"], float)
        assert isinstance(entry["f1"], float)
        assert isinstance(entry["features_used"], list)
        assert set(entry["features_used"]) == {"feat_a", "feat_b"}

    def test_tmp_file_not_left_behind(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert not (tmp_path / "outputs" / "trace.tmp.json").exists()

    def test_outputs_dir_created_if_not_present(self, tmp_path):
        csv = make_csv(tmp_path / "data.csv")
        outputs = tmp_path / "outputs"
        assert not outputs.exists()
        subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"),
             "--dataset", str(csv), "--target", "label"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert outputs.is_dir()
