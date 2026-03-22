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


class FeatureShapEntry(BaseModel):
    feature_name: str
    mean_abs_shap: float
    rank: int


class ShapSummary(BaseModel):
    ranked_features: list[FeatureShapEntry]
    top_3_summary: str


class ReasoningOutput(BaseModel):
    hypothesis: str
    feature_name: str
    transformation_code: str
    decision_rationale: str


class IterationRecord(BaseModel):
    iteration: int
    hypothesis: str
    feature_name: str
    transformation_code: str
    auc_before: float
    auc_after: float
    auc_delta: float
    shap_summary: ShapSummary
    decision: str   # "kept" | "discarded" | "error"
    error_message: str | None
    status: str     # "completed" | "failed"


class ExecuteResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    new_columns: list[str]
    error_message: str | None
    output_df: object | None  # pd.DataFrame | None
