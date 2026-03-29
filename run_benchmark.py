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
_EFFICIENCY_KEYWORDS = ("efficiency", "ratio", "sqft", "area", "per_age", "age_ratio")
_LOCATION_KEYWORDS = ("distance", "location", "decay", "center", "proximity", "dist")


def _has_keyword(name: str, keywords: tuple) -> bool:
    n = name.lower()
    return any(kw in n for kw in keywords)


def _run_dataset(dataset_path: str, target_col: str, max_iter: int = 5,
                 task_type: str | None = None):
    trace = AgentLoop().run(
        dataset_path=dataset_path,
        target_col=target_col,
        max_iter=max_iter,
        task_type=task_type,
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


def _regression_section(title: str, formatted) -> list[str]:
    features = [f.name for f in formatted.kept_features]
    # For regression, baseline_auc/final_auc hold RMSE (primary_metric) via property alias
    baseline_rmse = formatted.baseline_auc
    final_rmse = formatted.final_auc
    if baseline_rmse > 0:
        improvement_pct = (baseline_rmse - final_rmse) / baseline_rmse * 100
    else:
        improvement_pct = 0.0
    has_efficiency = any(_has_keyword(n, _EFFICIENCY_KEYWORDS) for n in features)
    has_location = any(_has_keyword(n, _LOCATION_KEYWORDS) for n in features)
    found = "Yes" if (has_efficiency and has_location) else "No"
    return [
        f"### {title}",
        f"- Baseline RMSE: {baseline_rmse:.2f} | "
        f"Final RMSE: {final_rmse:.2f} | "
        f"Improvement: {improvement_pct:.1f}%",
        f"- Features discovered: {features}",
        f"- Hidden signal found: {found}",
    ]


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Running agent on Synthetic Churn dataset...")
    churn_fmt = _run_dataset(str(DATA_DIR / "synthetic_churn.csv"), "churn")

    print("Running agent on Synthetic Regression dataset...")
    reg_fmt = _run_dataset(
        str(DATA_DIR / "synthetic_regression.csv"),
        "price",
        task_type="regression",
    )

    report_lines = (
        [
            "## Benchmark Results",
            "",
            "### UCI Bank Marketing",
            "- Baseline AUC: N/A | Final AUC: N/A | Lift: N/A",
            "- Features discovered: []",
            "  _(UCI dataset not included in this repository)_",
            "",
        ]
        + _section("Synthetic Churn", churn_fmt, hidden_signal_check=True)
        + [""]
        + _regression_section("Synthetic Regression (House Price)", reg_fmt)
        + [""]
    )

    report = "\n".join(report_lines)
    report_path = OUTPUTS_DIR / "benchmark_report.md"
    report_path.write_text(report)
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
