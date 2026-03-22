import argparse
import json
import os
import pathlib
import sys

from agent.data_loader import DatasetLoader
from tools.evaluate import EvaluateTool
from tools.profile import ProfileTool

OUTPUTS_DIR = pathlib.Path("outputs")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering agent")
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations (default: 5)")
    args = parser.parse_args()

    original_df, working_df = DatasetLoader().load(args.dataset, args.target)

    ProfileTool().profile(working_df, args.target)

    result = EvaluateTool().evaluate(working_df, args.target)

    trace_entry = {
        "iteration": 0,
        "status": "baseline",
        "auc": result.auc,
        "f1": result.f1,
        "features_used": result.feature_names,
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    trace_path = OUTPUTS_DIR / "trace.json"
    tmp_path = OUTPUTS_DIR / "trace.json.tmp"
    tmp_path.write_text(json.dumps([trace_entry], indent=2))
    tmp_path.replace(trace_path)

    print(f"Baseline AUC: {result.auc:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
