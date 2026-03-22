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


class ExecuteResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    new_columns: list[str]
    error_message: str | None
    output_df: object | None  # pd.DataFrame | None
