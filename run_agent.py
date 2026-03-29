import argparse
import csv
import pathlib
import sys

from agent.loop import AgentLoop
from agent.output_formatter import OutputFormatter
from tools.profile import ProfileTool


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Feature engineering agent")
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations (default: 5)")
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default=None,
        help="Task type (default: auto-detect from target column)",
    )
    args = parser.parse_args()

    trace = AgentLoop().run(
        dataset_path=args.dataset,
        target_col=args.target,
        max_iter=args.max_iter,
        task_type=args.task_type,
    )

    import pandas as pd
    raw_df = pd.read_csv(args.dataset)
    profile = ProfileTool().profile(raw_df, args.target)
    formatted = OutputFormatter().format(trace, profile)

    is_regression = trace.task_type.value == "regression"
    n_iter = len(trace.iterations)
    k = len(formatted.kept_features)

    if is_regression:
        improvement = trace.baseline_metric - trace.final_metric
        summary = (
            f"Agent ran {n_iter} iterations on {args.target}. "
            f"Final RMSE: {trace.final_metric:.4f} "
            f"(baseline: {trace.baseline_metric:.4f}, "
            f"improvement: {improvement:+.4f})"
        )
    else:
        lift = trace.final_metric - trace.baseline_metric
        summary = (
            f"Agent ran {n_iter} iterations on {args.target}. "
            f"Baseline AUC: {trace.baseline_metric:.4f}. "
            f"Final AUC: {trace.final_metric:.4f}. "
            f"Lift: {lift:.4f}."
        )

    print(summary)
    print(f"{k} features kept:")
    for idx, feat in enumerate(formatted.kept_features, start=1):
        print(f"{idx}. {feat.name}: {feat.hypothesis} (SHAP contribution: {feat.mean_abs_shap:.4f})")

    outputs_dir = pathlib.Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    csv_path = outputs_dir / "final_features.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "hypothesis", "mean_abs_shap", "auc_delta", "decision", "transformation_code"],
        )
        writer.writeheader()
        for feat in formatted.kept_features:
            writer.writerow({
                "name": feat.name,
                "hypothesis": feat.hypothesis,
                "mean_abs_shap": feat.mean_abs_shap,
                "auc_delta": feat.auc_delta,
                "decision": feat.decision,
                "transformation_code": feat.transformation_code,
            })

    sys.exit(0)


if __name__ == "__main__":
    main()
