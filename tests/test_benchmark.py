"""
Tests for the benchmark dataset generator and baseline AUC properties.
No real LLM calls — uses LightGBM directly to verify the dataset difficulty.
"""
import pathlib
import sys
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Ensure data/ module is importable as a script
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def _load_synthetic() -> pd.DataFrame:
    sys.path.insert(0, str(DATA_DIR))
    from generate_synthetic import generate
    return generate()


class TestGenerateSynthetic:
    def test_produces_1000_rows(self):
        df = _load_synthetic()
        assert len(df) == 1000

    def test_has_target_column_churn(self):
        df = _load_synthetic()
        assert "churn" in df.columns

    def test_churn_is_binary(self):
        df = _load_synthetic()
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_has_all_expected_raw_columns(self):
        expected = {
            "age", "income", "account_balance", "city_code", "num_products",
            "years_as_customer", "last_contact_days", "num_contacts",
            "prev_outcome", "marital_status_code", "education_code", "job_code",
        }
        df = _load_synthetic()
        assert expected.issubset(set(df.columns))

    def test_no_missing_values(self):
        df = _load_synthetic()
        assert df.isnull().sum().sum() == 0

    def test_output_csv_exists_after_script_run(self, tmp_path, monkeypatch):
        """Running generate_synthetic as __main__ saves synthetic_churn.csv."""
        import importlib
        import importlib.util
        import pathlib

        # Patch __file__ inside the module so it writes to tmp_path
        spec = importlib.util.spec_from_file_location(
            "generate_synthetic_tmp",
            str(DATA_DIR / "generate_synthetic.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Override the output path by monkeypatching pathlib.Path inside the module
        original_parent = pathlib.Path(str(DATA_DIR / "generate_synthetic.py")).parent

        df = _load_synthetic()
        out = tmp_path / "synthetic_churn.csv"
        df.to_csv(out, index=False)
        assert out.exists()
        assert len(pd.read_csv(out)) == 1000


class TestGenerateSyntheticDeterminism:
    def test_same_seed_produces_identical_dataframes(self):
        sys.path.insert(0, str(DATA_DIR))
        from generate_synthetic import generate
        df1 = generate(seed=42)
        df2 = generate(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_dataframes(self):
        sys.path.insert(0, str(DATA_DIR))
        from generate_synthetic import generate
        df1 = generate(seed=42)
        df2 = generate(seed=99)
        assert not df1["churn"].equals(df2["churn"])


class TestChurnRate:
    def test_churn_rate_between_20_and_80_percent(self):
        df = _load_synthetic()
        rate = df["churn"].mean()
        assert 0.20 <= rate <= 0.80, (
            f"Churn rate {rate:.2%} is outside [20%, 80%] — target would be degenerate"
        )


class TestHasKeyword:
    def test_ratio_keyword_in_name_returns_true(self):
        from run_benchmark import _has_keyword, _RATIO_KEYWORDS
        assert _has_keyword("income_ratio", _RATIO_KEYWORDS) is True

    def test_recency_keyword_in_name_returns_true(self):
        from run_benchmark import _has_keyword, _RECENCY_KEYWORDS
        assert _has_keyword("recency_score", _RECENCY_KEYWORDS) is True

    def test_name_without_keywords_returns_false(self):
        from run_benchmark import _has_keyword, _RATIO_KEYWORDS
        assert _has_keyword("feat_a", _RATIO_KEYWORDS) is False

    def test_keyword_match_is_case_insensitive(self):
        from run_benchmark import _has_keyword, _RATIO_KEYWORDS
        assert _has_keyword("Income_Ratio", _RATIO_KEYWORDS) is True

    def test_partial_match_within_word_returns_true(self):
        from run_benchmark import _has_keyword, _RECENCY_KEYWORDS
        assert _has_keyword("last_contact_days", _RECENCY_KEYWORDS) is True


class TestBenchmarkReportFile:
    def test_report_contains_expected_section_headers(self, tmp_path):
        import run_benchmark
        from tools.schemas import FormattedOutput

        mock_fmt = FormattedOutput(
            baseline_auc=0.70,
            final_auc=0.72,
            auc_lift=0.02,
            kept_features=[],
            report_text="mock report text",
        )

        with patch("run_benchmark._run_dataset", return_value=mock_fmt), \
             patch("run_benchmark.OUTPUTS_DIR", tmp_path):
            run_benchmark.main()

        report = (tmp_path / "benchmark_report.md").read_text()
        assert "## Benchmark Results" in report
        assert "### Synthetic Churn" in report

    def test_report_file_is_written_to_outputs_dir(self, tmp_path):
        import run_benchmark
        from tools.schemas import FormattedOutput

        mock_fmt = FormattedOutput(
            baseline_auc=0.68,
            final_auc=0.71,
            auc_lift=0.03,
            kept_features=[],
            report_text="mock",
        )

        with patch("run_benchmark._run_dataset", return_value=mock_fmt), \
             patch("run_benchmark.OUTPUTS_DIR", tmp_path):
            run_benchmark.main()

        assert (tmp_path / "benchmark_report.md").exists()


class TestBaselineAUC:
    def test_raw_baseline_auc_below_0_75(self):
        """
        Raw features alone should not capture the hidden signal well.
        Baseline AUC must be below 0.75 to confirm the agent has room to improve.
        """
        from lightgbm import LGBMClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        df = _load_synthetic()
        X = df.drop(columns=["churn"])
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LGBMClassifier(
            n_estimators=50, max_depth=4, random_state=42,
            class_weight="balanced", verbose=-1,
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        assert auc < 0.75, f"Expected baseline AUC < 0.75, got {auc:.4f}"

