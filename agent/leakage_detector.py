import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from agent.logger import get_logger
from tools.schemas import LeakageResult, TaskType

logger = get_logger("agent.leakage")

_CORR_THRESHOLD = 0.95
_MI_THRESHOLD = 0.9


class LeakageDetector:
    def is_leaking(
        self,
        feature_series: pd.Series,
        target_series: pd.Series,
        feature_name: str,
        target_col: str,
        task_type: TaskType = TaskType.classification,
    ) -> LeakageResult:
        # Check 1: feature name contains target column name
        if target_col.lower() in feature_name.lower():
            result = LeakageResult(
                is_leaking=True,
                reason=(
                    f"Feature name '{feature_name}' contains "
                    f"target column name '{target_col}'."
                ),
            )
            logger.debug(f"Leakage check '{feature_name}': True — {result.reason}")
            return result

        # Check 2: Pearson correlation abs > 0.95
        try:
            corr, _ = stats.pearsonr(feature_series, target_series)
            if abs(corr) > _CORR_THRESHOLD:
                result = LeakageResult(
                    is_leaking=True,
                    reason=(
                        f"Pearson correlation with target is {corr:.4f} "
                        f"(threshold: {_CORR_THRESHOLD})."
                    ),
                )
                logger.debug(f"Leakage check '{feature_name}': True — {result.reason}")
                return result
        except Exception:
            pass

        # Check 3: Mutual information > 0.9
        try:
            X = feature_series.values.reshape(-1, 1)
            if task_type == TaskType.regression:
                mi = mutual_info_regression(X, target_series, random_state=42)[0]
            else:
                mi = mutual_info_classif(X, target_series, random_state=42)[0]
            if mi > _MI_THRESHOLD:
                result = LeakageResult(
                    is_leaking=True,
                    reason=(
                        f"Mutual information with target is {mi:.4f} "
                        f"(threshold: {_MI_THRESHOLD})."
                    ),
                )
                logger.debug(f"Leakage check '{feature_name}': True — {result.reason}")
                return result
        except Exception:
            pass

        result = LeakageResult(is_leaking=False, reason=None)
        logger.debug(f"Leakage check '{feature_name}': {result.is_leaking} — passed all checks")
        return result
