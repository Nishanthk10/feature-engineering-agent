from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


class EvaluationResult(BaseModel):
    primary_metric: float
    secondary_metric: float
    shap_values: dict[str, float]
    feature_names: list[str]
    task_type: TaskType = TaskType.classification

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        if isinstance(data, dict):
            if "auc" in data and "primary_metric" not in data:
                data["primary_metric"] = data.pop("auc")
            if "f1" in data and "secondary_metric" not in data:
                data["secondary_metric"] = data.pop("f1")
        return data

    @property
    def auc(self) -> float:
        return self.primary_metric

    @property
    def f1(self) -> float:
        return self.secondary_metric


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    target_col: str
    feature_cols: list[str]
    missing_rate: dict[str, float]
    dtypes: dict[str, str]
    sample_rows: list[dict] = []
    data_dictionary: dict[str, str] = {}


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


class AgentTrace(BaseModel):
    baseline_metric: float
    iterations: list["IterationRecord"]
    final_feature_set: list[str]
    final_metric: float
    task_type: TaskType = TaskType.classification

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        if isinstance(data, dict):
            if "baseline_auc" in data and "baseline_metric" not in data:
                data["baseline_metric"] = data.pop("baseline_auc")
            if "final_auc" in data and "final_metric" not in data:
                data["final_metric"] = data.pop("final_auc")
        return data

    @property
    def baseline_auc(self) -> float:
        return self.baseline_metric

    @property
    def final_auc(self) -> float:
        return self.final_metric


class FeatureCandidate(BaseModel):
    name: str
    transformation_code: str
    hypothesis: str
    mean_abs_shap: float
    auc_delta: float
    decision: str  # "kept" | "discarded" | "error"

    @field_validator("name", "transformation_code", "hypothesis")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v


class FormattedOutput(BaseModel):
    baseline_auc: float
    final_auc: float
    auc_lift: float
    kept_features: list["FeatureCandidate"]
    report_text: str


class LeakageResult(BaseModel):
    is_leaking: bool
    reason: str | None


class ExecuteResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    new_columns: list[str]
    error_message: str | None
    output_df: object | None  # pd.DataFrame | None
