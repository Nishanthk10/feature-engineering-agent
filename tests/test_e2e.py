"""
End-to-end tests using real LLM calls and real filesystem writes.
No mocking. Run with: pytest tests/test_e2e.py -m e2e -v

Prerequisites:
    python data/generate_synthetic.py   # generates the CSV fixtures
    Set LLM_PROVIDER and the matching API key in .env
"""
import json
import pathlib
from unittest.mock import patch

import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


@pytest.mark.e2e
def test_classification_auc_lift(tmp_path):
    """Agent must improve AUC on the synthetic churn dataset."""
    dataset = DATA_DIR / "synthetic_churn.csv"
    if not dataset.exists():
        pytest.skip("synthetic_churn.csv not found — run: python data/generate_synthetic.py")

    from agent.loop import AgentLoop

    with patch("agent.loop.OUTPUTS_DIR", tmp_path / "outputs"):
        trace = AgentLoop().run(
            dataset_path=str(dataset),
            target_col="churn",
            max_iter=5,
        )

    assert trace.final_metric > trace.baseline_metric, (
        f"Expected AUC lift: final {trace.final_metric:.4f} <= baseline {trace.baseline_metric:.4f}"
    )
    assert trace.task_type.value == "classification"
    assert len(trace.iterations) >= 1

    trace_file = tmp_path / "outputs" / "trace.json"
    assert trace_file.exists(), "trace.json was not written"
    entries = json.loads(trace_file.read_text())
    assert len(entries) >= 2, "trace must contain baseline + at least 1 iteration"
    assert entries[0]["status"] == "baseline"
    assert entries[0]["task_type"] == "classification"


@pytest.mark.e2e
def test_regression_rmse_reduction(tmp_path):
    """Agent must reduce RMSE on the synthetic house-price dataset."""
    dataset = DATA_DIR / "synthetic_regression.csv"
    if not dataset.exists():
        pytest.skip("synthetic_regression.csv not found — run: python data/generate_synthetic.py")

    from agent.loop import AgentLoop

    with patch("agent.loop.OUTPUTS_DIR", tmp_path / "outputs"):
        trace = AgentLoop().run(
            dataset_path=str(dataset),
            target_col="price",
            max_iter=5,
            task_type="regression",
        )

    assert trace.final_metric < trace.baseline_metric, (
        f"Expected RMSE reduction: final {trace.final_metric:.2f} >= baseline {trace.baseline_metric:.2f}"
    )
    assert trace.task_type.value == "regression"
    assert len(trace.iterations) >= 1

    trace_file = tmp_path / "outputs" / "trace.json"
    assert trace_file.exists(), "trace.json was not written"
    entries = json.loads(trace_file.read_text())
    assert len(entries) >= 2, "trace must contain baseline + at least 1 iteration"
    assert entries[0]["status"] == "baseline"
    assert entries[0]["task_type"] == "regression"
