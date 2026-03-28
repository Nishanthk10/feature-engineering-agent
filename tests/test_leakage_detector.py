import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from agent.leakage_detector import LeakageDetector
from tools.schemas import LeakageResult


def _detector() -> LeakageDetector:
    return LeakageDetector()


class TestLeakageDetectorCorrelation:
    def test_feature_identical_to_target_is_leaking(self):
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        feature = target.copy()
        result = _detector().is_leaking(feature, target, "some_feat", "target")
        assert result.is_leaking is True

    def test_feature_perfectly_negatively_correlated_is_leaking(self):
        target = pd.Series([0.0, 1.0, 0.0, 1.0] * 25)
        feature = 1.0 - target
        result = _detector().is_leaking(feature, target, "some_feat", "target")
        assert result.is_leaking is True


class TestLeakageDetectorNameCheck:
    def test_feature_name_contains_target_col_is_leaking(self):
        rng = np.random.default_rng(0)
        target = pd.Series(rng.integers(0, 2, 100))
        feature = pd.Series(rng.random(100))
        result = _detector().is_leaking(feature, target, "target_squared", "target")
        assert result.is_leaking is True

    def test_feature_name_contains_target_col_case_insensitive(self):
        rng = np.random.default_rng(0)
        target = pd.Series(rng.integers(0, 2, 100))
        feature = pd.Series(rng.random(100))
        result = _detector().is_leaking(feature, target, "TARGET_ratio", "target")
        assert result.is_leaking is True

    def test_feature_name_not_containing_target_col_is_not_flagged_by_name(self):
        rng = np.random.default_rng(0)
        target = pd.Series(rng.integers(0, 2, 100))
        feature = pd.Series(rng.random(100))
        result = _detector().is_leaking(feature, target, "income_ratio", "target")
        # name check alone should not trigger; uncorrelated data means no leakage
        assert result.is_leaking is False


class TestLeakageDetectorCleanFeature:
    def test_uncorrelated_feature_is_not_leaking(self):
        rng = np.random.default_rng(42)
        target = pd.Series(rng.integers(0, 2, 200))
        feature = pd.Series(rng.random(200))
        result = _detector().is_leaking(feature, target, "random_noise", "target")
        assert result.is_leaking is False

    def test_result_is_leakage_result_instance(self):
        rng = np.random.default_rng(1)
        target = pd.Series(rng.integers(0, 2, 50))
        feature = pd.Series(rng.random(50))
        result = _detector().is_leaking(feature, target, "feat", "target")
        assert isinstance(result, LeakageResult)


class TestLeakageDetectorMIBranch:
    def test_high_mi_is_caught_when_name_and_correlation_pass(self):
        """Third branch (MI > 0.9) exercised after name and Pearson checks both pass."""
        rng = np.random.default_rng(5)
        target = pd.Series(rng.integers(0, 2, 300))
        feature = pd.Series(rng.random(300))  # random: low correlation, safe name

        with patch(
            "agent.leakage_detector.mutual_info_classif",
            return_value=np.array([0.95]),
        ):
            result = _detector().is_leaking(feature, target, "safe_feature", "label")

        assert result.is_leaking is True

    def test_mi_branch_not_reached_when_name_check_fires_first(self):
        """Name check fires before MI is ever computed."""
        rng = np.random.default_rng(5)
        target = pd.Series(rng.integers(0, 2, 100))
        feature = pd.Series(rng.random(100))

        with patch(
            "agent.leakage_detector.mutual_info_classif",
        ) as mi_mock:
            _detector().is_leaking(feature, target, "target_derived", "target")

        mi_mock.assert_not_called()


class TestLeakageDetectorEdgeCases:
    def test_constant_feature_does_not_raise(self):
        """Constant feature causes undefined Pearson correlation; should not crash."""
        target = pd.Series([0, 1] * 50)
        feature = pd.Series([1.0] * 100)
        result = _detector().is_leaking(feature, target, "constant_feat", "label")
        assert isinstance(result, LeakageResult)

    def test_constant_feature_is_not_flagged_as_leaking(self):
        """A constant feature has no information about the target."""
        target = pd.Series([0, 1] * 50)
        feature = pd.Series([3.14] * 100)
        result = _detector().is_leaking(feature, target, "flat_feat", "label")
        assert result.is_leaking is False


class TestLeakageDetectorReasonField:
    def test_reason_set_when_leaking_by_name(self):
        rng = np.random.default_rng(0)
        target = pd.Series(rng.integers(0, 2, 100))
        feature = pd.Series(rng.random(100))
        result = _detector().is_leaking(feature, target, "target_derived", "target")
        assert result.is_leaking is True
        assert result.reason is not None
        assert len(result.reason) > 0

    def test_reason_set_when_leaking_by_correlation(self):
        target = pd.Series([0, 1] * 50, dtype=float)
        feature = target * 2.0  # perfectly correlated, different name
        result = _detector().is_leaking(feature, target, "double_feat", "label")
        assert result.is_leaking is True
        assert result.reason is not None

    def test_reason_is_none_when_not_leaking(self):
        rng = np.random.default_rng(7)
        target = pd.Series(rng.integers(0, 2, 200))
        feature = pd.Series(rng.random(200))
        result = _detector().is_leaking(feature, target, "safe_feat", "target")
        assert result.is_leaking is False
        assert result.reason is None
