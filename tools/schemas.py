from pydantic import BaseModel


class EvaluationResult(BaseModel):
    auc: float
    f1: float
    shap_values: dict[str, float]
    feature_names: list[str]
