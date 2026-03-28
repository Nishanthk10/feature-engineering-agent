import json

import pandas as pd
from fastmcp import FastMCP

from agent.data_loader import DatasetLoader
from tools.evaluate import EvaluateTool
from tools.execute import ExecuteTool
from tools.profile import ProfileTool
from tools.schemas import EvaluationResult
from tools.shap_tool import ShapTool

mcp = FastMCP("feature-agent")


@mcp.tool()
def profile_dataset(csv_path: str, target_col: str) -> dict:
    """Profile a CSV dataset and return column statistics."""
    _, working_df = DatasetLoader().load(csv_path, target_col)
    profile = ProfileTool().profile(working_df, target_col)
    return profile.model_dump()


@mcp.tool()
def execute_feature_code(df_json: str, code: str) -> dict:
    """Execute feature engineering code in the sandbox and return the result."""
    df = pd.read_json(df_json)
    result = ExecuteTool().execute(df, code)
    output = result.model_dump()
    output["output_df"] = result.output_df.to_json() if result.output_df is not None else None
    return output


@mcp.tool()
def evaluate_features(df_json: str, target_col: str) -> dict:
    """Train LightGBM on the dataframe and return AUC, F1, and SHAP values."""
    df = pd.read_json(df_json)
    result = EvaluateTool().evaluate(df, target_col)
    return result.model_dump()


@mcp.tool()
def get_shap_values(eval_result_json: str) -> dict:
    """Rank features by SHAP importance and return a summary for the LLM."""
    try:
        data = json.loads(eval_result_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for eval_result_json: {exc}") from exc
    eval_result = EvaluationResult.model_validate(data)
    summary = ShapTool().format_for_llm(eval_result)
    return summary.model_dump()


if __name__ == "__main__":
    mcp.run()
