import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from agent.logger import get_logger
from tools.schemas import (
    DatasetProfile,
    IterationRecord,
    ReasoningOutput,
    ShapSummary,
    TaskType,
)

logger = get_logger("agent.llm")

LLM_TIMEOUT = 60  # seconds

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


def _build_shap_context(shap_summary: ShapSummary, iteration_number: int) -> tuple[str, str]:
    """Return (shap_context, tier_instruction) based on the current iteration."""
    ranked = shap_summary.ranked_features
    n = len(ranked)

    if n <= 3:
        shap_context = "All features by importance:\n" + "\n".join(
            f"  {i+1}. {f.feature_name} (shap={f.mean_abs_shap:.4f})"
            for i, f in enumerate(ranked)
        )
        tier_instruction = ""

    elif iteration_number <= 2:
        top = ranked[:3]
        shap_context = "Top features by importance:\n" + "\n".join(
            f"  {i+1}. {f.feature_name} (shap={f.mean_abs_shap:.4f})"
            for i, f in enumerate(top)
        )
        tier_instruction = "Focus on the top features. Establish strong signal first."

    elif iteration_number <= 4:
        top = ranked[0]
        mid = ranked[n // 2]
        shap_context = (
            f"Strong feature: {top.feature_name} (shap={top.mean_abs_shap:.4f})\n"
            f"Mid-tier feature: {mid.feature_name} (shap={mid.mean_abs_shap:.4f})\n"
            f"All features: " + ", ".join(f.feature_name for f in ranked)
        )
        tier_instruction = (
            f"The top features are already well-exploited. "
            f"Try an interaction between '{top.feature_name}' (strong) "
            f"and '{mid.feature_name}' (mid-tier). "
            f"Cross-tier interactions often reveal non-linear relationships "
            f"the model missed when evaluating features independently."
        )

    else:
        top = ranked[0]
        bottom = ranked[-1]
        second_bottom = ranked[-2] if n >= 2 else ranked[-1]
        shap_context = (
            f"Strong feature: {top.feature_name} (shap={top.mean_abs_shap:.4f})\n"
            f"Weak features: {bottom.feature_name} (shap={bottom.mean_abs_shap:.4f}), "
            f"{second_bottom.feature_name} (shap={second_bottom.mean_abs_shap:.4f})\n"
            f"All features: " + ", ".join(f.feature_name for f in ranked)
        )
        tier_instruction = (
            f"Top features are exhausted. Explore combinations of weak signals. "
            f"Try: '{top.feature_name}' × '{bottom.feature_name}', or an interaction "
            f"between the two weakest features. Sometimes weak features encode "
            f"hidden segment information that only surfaces in combination."
        )

    return shap_context, tier_instruction


def _build_user_prompt(
    profile: DatasetProfile,
    shap_summary: ShapSummary,
    iteration_history: list[IterationRecord],
    current_features: list[str],
    iteration_number: int = 1,
) -> str:
    dtypes_str = ", ".join(f"{col}: {dtype}" for col, dtype in profile.dtypes.items()
                           if col != profile.target_col)

    history_lines: list[str] = []
    for rec in iteration_history[-3:]:
        history_lines.append(
            f"  - Iteration {rec.iteration}: \"{rec.hypothesis}\" → AUC delta {rec.auc_delta:+.4f}"
        )
    history_str = "\n".join(history_lines) if history_lines else "  (no prior iterations)"

    dict_section = ""
    if profile.data_dictionary:
        dict_section = "\nData dictionary (user-provided):\n"
        for col, desc in profile.data_dictionary.items():
            dict_section += f"  {col}: {desc}\n"

    sample_section = ""
    if profile.sample_rows:
        sample_section = "\nSample data (5 rows):\n"
        for col in profile.feature_cols[:10]:
            values = [str(row.get(col, "")) for row in profile.sample_rows]
            sample_section += f"  {col}: [{', '.join(values)}]\n"

    shap_context, tier_instruction = _build_shap_context(shap_summary, iteration_number)
    tier_section = f"\nSHAP exploration directive:\n  {tier_instruction}\n" if tier_instruction else ""

    return f"""\
Dataset profile:
  rows: {profile.row_count}
  feature columns: {profile.feature_cols}
  dtypes: {dtypes_str}

Current engineered features: {current_features}
{dict_section}{sample_section}
SHAP importance summary:
{shap_context}
{tier_section}
Recent iteration history:
{history_str}

Propose the single most promising next feature to engineer.\
"""


class LLMClient:
    def _call_provider(self, system: str, user: str) -> str:
        provider = os.environ.get("LLM_PROVIDER", "gemini")

        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            response = client.models.generate_content(
                model=model,
                contents=system + "\n\n" + user,
            )
            return response.text

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text

        elif provider == "huggingface":
            from huggingface_hub import InferenceClient
            model = os.environ.get("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            client = InferenceClient(
                model=model,
                token=os.environ.get("HUGGINGFACE_API_KEY"),
            )
            response = client.text_generation(
                system + "\n\n" + user,
                max_new_tokens=1000,
            )
            return response

        elif provider == "nvidia":
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.environ.get("NVIDIA_API_KEY"),
            )
            model = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2")
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=1,
                top_p=0.95,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True,
            )
            content_parts: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
            return "".join(content_parts)

        elif provider == "openrouter":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
                    "X-Title": os.environ.get("OPENROUTER_SITE_NAME", "feature-agent"),
                },
            )
            model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider}. "
                "Supported: gemini, openai, anthropic, huggingface, openrouter"
            )

    def complete(self, system: str, user: str) -> str:
        provider = os.environ.get("LLM_PROVIDER", "gemini")
        logger.debug(f"LLM call starting (timeout={LLM_TIMEOUT}s, provider={provider})")
        last_exc: Exception | None = None
        for attempt in range(1, 4):  # up to 3 attempts
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._call_provider, system, user)
                try:
                    result = future.result(timeout=LLM_TIMEOUT)
                    logger.debug(f"LLM call returned successfully (attempt {attempt})")
                    return result
                except FuturesTimeout:
                    future.cancel()
                    raise TimeoutError(f"LLM API call timed out after {LLM_TIMEOUT}s")
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if attempt < 3 and ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str):
                        wait = 5 * attempt
                        logger.warning(f"LLM attempt {attempt} failed ({err_str[:120]}), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
        raise last_exc  # type: ignore


class LLMReasoner:
    def __init__(self) -> None:
        self._client = LLMClient()

    def reason(
        self,
        profile: DatasetProfile,
        shap_summary: ShapSummary,
        iteration_history: list[IterationRecord],
        current_features: list[str],
        task_type: TaskType = TaskType.classification,
        iteration_number: int = 1,
    ) -> ReasoningOutput:
        if task_type == TaskType.regression:
            task_note = (
                "\nTask type: regression. "
                "Primary metric: RMSE (lower is better). "
                "Propose features that reduce prediction error."
            )
        else:
            task_note = (
                "\nTask type: classification. "
                "Primary metric: AUC (higher is better). "
                "Propose features that improve class discrimination."
            )
        system = SYSTEM_PROMPT + task_note

        user_prompt = _build_user_prompt(profile, shap_summary, iteration_history, current_features, iteration_number)

        try:
            raw = self._client.complete(system, user_prompt)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}", exc_info=True)
            raise

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip markdown code fences: ```json ... ``` or ``` ... ```
            cleaned = cleaned.split("\n", 1)[-1]  # drop the opening fence line
            if cleaned.endswith("```"):
                cleaned = cleaned[: cleaned.rfind("```")]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"LLM returned invalid JSON. Raw response: {raw[:500]}", exc_info=True)
            raise ValueError(
                f"LLM returned invalid JSON: {exc}\nRaw response:\n{raw}"
            ) from exc

        output = ReasoningOutput.model_validate(data)
        logger.info(f"Hypothesis: {output.hypothesis[:120]}...")
        return output
