from pydantic import BaseModel


class EvaluationResult(BaseModel):
    auc: float
    f1: float
    shap_values: dict[str, float]
    feature_names: list[str]


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    target_col: str
    feature_cols: list[str]
    missing_rate: dict[str, float]
    dtypes: dict[str, str]
