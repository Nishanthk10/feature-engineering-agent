import json
import os

import anthropic

from tools.schemas import (
    DatasetProfile,
    IterationRecord,
    ReasoningOutput,
    ShapSummary,
)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are an expert data scientist specialising in feature engineering.

Always respond with a JSON object containing exactly these keys:
  - hypothesis: plain English explanation of why this feature should improve model performance
  - feature_name: a short snake_case name for the new feature column
  - transformation_code: valid Python/pandas code that adds the feature to df
  - decision_rationale: why this is the best next feature to try given the iteration history

Constraints for transformation_code:
  - Use only "df" as the variable name for the DataFrame
  - Add exactly one new column, named exactly <feature_name>, to df
  - Allowed imports: pandas, numpy, scipy.stats, sklearn.preprocessing
  - Do not read from or write to disk
  - Do not reference the target column

Respond with JSON only. No markdown fences, no explanation outside the JSON object.\
"""


def _build_user_prompt(
    profile: DatasetProfile,
    shap_summary: ShapSummary,
    iteration_history: list[IterationRecord],
    current_features: list[str],
) -> str:
    dtypes_str = ", ".join(f"{col}: {dtype}" for col, dtype in profile.dtypes.items()
                           if col != profile.target_col)

    history_lines: list[str] = []
    for rec in iteration_history[-3:]:
        history_lines.append(
            f"  - Iteration {rec.iteration}: \"{rec.hypothesis}\" → AUC delta {rec.auc_delta:+.4f}"
        )
    history_str = "\n".join(history_lines) if history_lines else "  (no prior iterations)"

    return f"""\
Dataset profile:
  rows: {profile.row_count}
  feature columns: {profile.feature_cols}
  dtypes: {dtypes_str}

Current engineered features: {current_features}

SHAP importance summary:
  {shap_summary.top_3_summary}

Recent iteration history:
{history_str}

Propose the single most promising next feature to engineer.\
"""


class LLMReasoner:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def reason(
        self,
        profile: DatasetProfile,
        shap_summary: ShapSummary,
        iteration_history: list[IterationRecord],
        current_features: list[str],
    ) -> ReasoningOutput:
        user_prompt = _build_user_prompt(profile, shap_summary, iteration_history, current_features)

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = message.content[0].text

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}\nRaw response:\n{raw}"
            ) from exc

        return ReasoningOutput.model_validate(data)
