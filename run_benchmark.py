"""
run_benchmark.py — runs the feature engineering agent on benchmark datasets
and writes outputs/benchmark_report.md.
"""
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.loop import AgentLoop
from agent.output_formatter import OutputFormatter
from tools.profile import ProfileTool

import pandas as pd

OUTPUTS_DIR = pathlib.Path("outputs")
DATA_DIR = pathlib.Path("data")

_RATIO_KEYWORDS = ("ratio", "rate", "div", "frac", "per")
_RECENCY_KEYWORDS = ("recency", "decay", "contact", "recent", "days")


def _has_keyword(name: str, keywords: tuple) -> bool:
    n = name.lower()
    return any(kw in n for kw in keywords)


def _run_dataset(dataset_path: str, target_col: str, max_iter: int = 5):
    trace = AgentLoop().run(
        dataset_path=dataset_path,
        target_col=target_col,
        max_iter=max_iter,
    )
    raw_df = pd.read_csv(dataset_path)
    profile = ProfileTool().profile(raw_df, target_col)
    formatted = OutputFormatter().format(trace, profile)
    return formatted


def _section(title: str, formatted, hidden_signal_check: bool = False) -> list[str]:
    features = [f.name for f in formatted.kept_features]
    lines = [
        f"### {title}",
        f"- Baseline AUC: {formatted.baseline_auc:.4f} | "
        f"Final AUC: {formatted.final_auc:.4f} | "
        f"Lift: {formatted.auc_lift:.4f}",
        f"- Features discovered: {features}",
    ]
    if hidden_signal_check:
        has_ratio = any(_has_keyword(n, _RATIO_KEYWORDS) for n in features)
        has_recency = any(_has_keyword(n, _RECENCY_KEYWORDS) for n in features)
        found = "Yes" if (has_ratio and has_recency) else "No"
        lines.append(f"- Hidden signal found: {found}")
    return lines


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Running agent on Synthetic Churn dataset...")
    churn_fmt = _run_dataset(str(DATA_DIR / "synthetic_churn.csv"), "churn")

    report_lines = [
        "## Benchmark Results",
        "",
        "### UCI Bank Marketing",
        "- Baseline AUC: N/A | Final AUC: N/A | Lift: N/A",
        "- Features discovered: []",
        "  _(UCI dataset not included in this repository)_",
        "",
    ] + _section("Synthetic Churn", churn_fmt, hidden_signal_check=True) + [""]

    report = "\n".join(report_lines)
    report_path = OUTPUTS_DIR / "benchmark_report.md"
    report_path.write_text(report)
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
