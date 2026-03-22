import argparse
import sys

from agent.loop import AgentLoop


def main():
    parser = argparse.ArgumentParser(description="Feature engineering agent")
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations (default: 5)")
    args = parser.parse_args()

    trace = AgentLoop().run(
        dataset_path=args.dataset,
        target_col=args.target,
        max_iter=args.max_iter,
    )

    lift = trace.final_auc - trace.baseline_auc
    print(f"Final AUC: {trace.final_auc:.4f} (baseline: {trace.baseline_auc:.4f}, lift: {lift:.4f})")
    sys.exit(0)


if __name__ == "__main__":
    main()
