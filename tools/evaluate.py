import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tools.schemas import EvaluationResult, TaskType

SHAP_SAMPLE_SIZE = 5_000
SHAP_SAMPLE_THRESHOLD = 50_000


class EvaluateTool:
    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: TaskType = TaskType.classification,
    ) -> EvaluationResult:
        feature_names = [c for c in df.columns if c != target_col]
        X = df[feature_names].copy()
        y = df[target_col].copy()

        if task_type == TaskType.regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = LGBMRegressor(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            primary_metric = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            secondary_metric = float(r2_score(y_test, y_pred))
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                class_weight="balanced",
                verbose=-1,
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            primary_metric = float(roc_auc_score(y_test, y_prob))
            secondary_metric = float(f1_score(y_test, y_pred, average="weighted"))

        shap_X = X_test
        if len(X_test) > SHAP_SAMPLE_THRESHOLD:
            shap_X = X_test.sample(SHAP_SAMPLE_SIZE, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(shap_X)

        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]

        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        shap_dict = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}

        return EvaluationResult(
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            shap_values=shap_dict,
            feature_names=feature_names,
            task_type=task_type,
        )
