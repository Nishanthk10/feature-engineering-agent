"""
Tests for api/main.py using FastAPI TestClient.
No real AgentLoop calls — agent is mocked where needed.
"""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, _state, _state_lock

STATIC_DIR = pathlib.Path(__file__).parent.parent / "static"


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global agent state before each test."""
    with _state_lock:
        _state["status"] = "idle"
        _state["iteration"] = 0
    yield


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestServeUI:
    def test_get_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_get_root_returns_html_content_type(self, client):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_get_root_contains_form(self, client):
        response = client.get("/")
        assert b"<form" in response.content


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_200(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_initial_status_is_idle(self, client):
        response = client.get("/status")
        assert response.json()["status"] == "idle"

    def test_initial_iteration_is_zero(self, client):
        response = client.get("/status")
        assert response.json()["iteration"] == 0

    def test_status_reflects_state_update(self, client):
        with _state_lock:
            _state["status"] = "running"
            _state["iteration"] = 3
        response = client.get("/status")
        data = response.json()
        assert data["status"] == "running"
        assert data["iteration"] == 3


# ---------------------------------------------------------------------------
# GET /trace
# ---------------------------------------------------------------------------

class TestGetTrace:
    def test_returns_empty_trace_when_no_file(self, client, tmp_path):
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace")
        assert response.status_code == 200
        body = response.json()
        assert body["trace"] == []
        assert "status" in body

    def test_returns_trace_contents_when_file_exists(self, client, tmp_path):
        trace_data = [{"iteration": 0, "status": "baseline", "auc": 0.70}]
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace")
        assert response.status_code == 200
        assert response.json()["trace"] == trace_data

    def test_returns_empty_trace_on_malformed_json(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text("not valid json {")
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace")
        assert response.json()["trace"] == []

    def test_status_field_is_idle_when_state_is_idle(self, client, tmp_path):
        with _state_lock:
            _state["status"] = "idle"
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace")
        assert response.json()["status"] == "idle"

    def test_status_field_is_running_when_state_is_running(self, client, tmp_path):
        with _state_lock:
            _state["status"] = "running"
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace")
        assert response.json()["status"] == "running"

    def test_status_field_is_complete_when_state_is_complete(self, client, tmp_path):
        with _state_lock:
            _state["status"] = "complete"
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace")
        assert response.json()["status"] == "complete"

    def test_polling_same_trace_twice_returns_identical_iteration_count(self, client, tmp_path):
        """
        The client-side renderedIterations counter relies on the server always
        returning the full trace (not a cursor/delta). Verify that two successive
        GET /trace calls with unchanged data return the same number of iterations,
        confirming the idempotency contract the counter depends on.
        """
        trace_data = [
            {"iteration": 0, "status": "baseline", "auc": 0.70},
            {"iteration": 1, "status": "completed", "decision": "kept",
             "auc_before": 0.70, "auc_after": 0.72, "auc_delta": 0.02,
             "feature_name": "feat_a", "hypothesis": "test"},
            {"iteration": 2, "status": "completed", "decision": "discarded",
             "auc_before": 0.72, "auc_after": 0.71, "auc_delta": -0.01,
             "feature_name": "feat_b", "hypothesis": "test2"},
        ]
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))

        with patch("api.main.TRACE_PATH", trace_file):
            first = client.get("/trace").json()["trace"]
            second = client.get("/trace").json()["trace"]

        assert len(first) == len(second) == 3
        # Non-baseline entries (what appendNewCards filters on) are stable
        non_baseline_first = [e for e in first if e.get("status") != "baseline"]
        non_baseline_second = [e for e in second if e.get("status") != "baseline"]
        assert len(non_baseline_first) == len(non_baseline_second) == 2


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------

class TestPostRun:
    def _csv_bytes(self) -> bytes:
        return b"feat_a,feat_b,target\n1,2,0\n3,4,1\n"

    def test_returns_200(self, client):
        with patch("api.main._run_agent"):
            response = client.post(
                "/run",
                data={"target_col": "target", "max_iter": "2"},
                files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
            )
        assert response.status_code == 200

    def test_returns_started_status(self, client):
        with patch("api.main._run_agent"):
            response = client.post(
                "/run",
                data={"target_col": "target", "max_iter": "2"},
                files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
            )
        assert response.json()["status"] == "started"

    def test_response_contains_message(self, client):
        with patch("api.main._run_agent"):
            response = client.post(
                "/run",
                data={"target_col": "target", "max_iter": "2"},
                files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
            )
        assert "message" in response.json()

    def test_upload_file_is_saved(self, client, tmp_path):
        upload_dest = tmp_path / "upload.csv"
        mock_thread = MagicMock()
        with patch("api.main.UPLOAD_PATH", upload_dest), \
             patch("api.main.threading.Thread", return_value=mock_thread):
            client.post(
                "/run",
                data={"target_col": "target", "max_iter": "1"},
                files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
            )
        assert upload_dest.exists()
        assert b"feat_a" in upload_dest.read_bytes()

    def test_missing_target_col_returns_422(self, client):
        response = client.post(
            "/run",
            data={"max_iter": "2"},
            files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
        )
        assert response.status_code == 422

    def test_missing_file_returns_422(self, client):
        response = client.post(
            "/run",
            data={"target_col": "target", "max_iter": "2"},
        )
        assert response.status_code == 422

    def test_omitting_max_iter_uses_default_and_succeeds(self, client):
        with patch("api.main._run_agent"):
            response = client.post(
                "/run",
                data={"target_col": "target"},
                files={"file": ("data.csv", self._csv_bytes(), "text/csv")},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "started"


# ---------------------------------------------------------------------------
# _run_agent exception handling
# ---------------------------------------------------------------------------

class TestRunAgentExceptionHandling:
    def test_exception_in_agent_loop_sets_status_complete(self):
        """AgentLoop.run() raising must not leave status stuck at 'running'."""
        from api.main import _run_agent

        with _state_lock:
            _state["status"] = "idle"

        with patch("api.main.AgentLoop") as mock_cls:
            mock_cls.return_value.run.side_effect = RuntimeError("boom")
            _run_agent("fake.csv", "target", 1)

        with _state_lock:
            assert _state["status"] == "complete"

    def test_exception_in_agent_loop_does_not_raise_to_caller(self):
        """Background thread must swallow the exception — never propagate."""
        from api.main import _run_agent

        with patch("api.main.AgentLoop") as mock_cls:
            mock_cls.return_value.run.side_effect = ValueError("unexpected")
            try:
                _run_agent("fake.csv", "target", 1)
            except Exception as exc:
                pytest.fail(f"_run_agent propagated exception: {exc}")


# ---------------------------------------------------------------------------
# GET /trace/view
# ---------------------------------------------------------------------------

_FIXTURE_TRACE = [
    {
        "iteration": 0,
        "status": "baseline",
        "auc": 0.70,
        "f1": 0.62,
        "primary_metric": 0.70,
        "secondary_metric": 0.62,
        "task_type": "classification",
        "features_used": ["feat_a", "feat_b"],
    },
    {
        "iteration": 1,
        "status": "completed",
        "hypothesis": "feat_a squared captures non-linear signal",
        "feature_name": "feat_a_sq",
        "transformation_code": "df['feat_a_sq'] = df['feat_a'] ** 2",
        "auc_before": 0.70,
        "auc_after": 0.73,
        "auc_delta": 0.03,
        "shap_summary": {
            "ranked_features": [
                {"feature_name": "feat_a_sq", "mean_abs_shap": 0.15, "rank": 1},
                {"feature_name": "feat_a", "mean_abs_shap": 0.10, "rank": 2},
            ],
            "top_3_summary": "feat_a_sq (0.150), feat_a (0.100)",
        },
        "decision": "kept",
        "error_message": None,
    },
]


class TestTraceView:
    def test_returns_200(self, client, tmp_path):
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace/view")
        assert response.status_code == 200

    def test_content_type_is_html(self, client, tmp_path):
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace/view")
        assert "text/html" in response.headers["content-type"]

    def test_no_trace_file_returns_html(self, client, tmp_path):
        with patch("api.main.TRACE_PATH", tmp_path / "nonexistent.json"):
            response = client.get("/trace/view")
        assert b"Agent Reasoning Trace" in response.content

    def test_fixture_trace_contains_baseline_heading(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"Baseline" in response.content

    def test_fixture_trace_contains_task_type(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"classification" in response.content

    def test_fixture_trace_contains_hypothesis(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"feat_a squared captures non-linear signal" in response.content

    def test_fixture_trace_contains_feature_name(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"feat_a_sq" in response.content

    def test_fixture_trace_contains_transformation_code(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"df[&#x27;feat_a_sq&#x27;] = df[&#x27;feat_a&#x27;] ** 2" in response.content

    def test_fixture_trace_decision_kept_present(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"kept" in response.content

    def test_fixture_trace_shap_summary_present(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"feat_a_sq (0.150)" in response.content

    def test_fixture_trace_summary_section_present(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(_FIXTURE_TRACE))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"Summary" in response.content

    def test_malformed_json_returns_html_gracefully(self, client, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text("not valid json {")
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert response.status_code == 200
        assert b"Agent Reasoning Trace" in response.content


# ---------------------------------------------------------------------------
# GET /trace/view — content variant tests
# ---------------------------------------------------------------------------

def _make_view_trace(iteration_overrides: dict) -> list[dict]:
    """Return a minimal 2-entry trace (baseline + 1 iteration) for targeted tests."""
    baseline = {
        "iteration": 0,
        "status": "baseline",
        "auc": 0.70,
        "primary_metric": 0.70,
        "secondary_metric": 0.62,
        "task_type": "classification",
        "features_used": ["feat_a"],
    }
    iteration = {
        "iteration": 1,
        "status": "completed",
        "hypothesis": "test hypothesis",
        "feature_name": "feat_new",
        "transformation_code": "df['feat_new'] = df['feat_a'] * 2",
        "auc_before": 0.70,
        "auc_after": 0.72,
        "auc_delta": 0.02,
        "shap_summary": {
            "ranked_features": [{"feature_name": "feat_new", "mean_abs_shap": 0.12, "rank": 1}],
            "top_3_summary": "feat_new (0.120)",
        },
        "decision": "kept",
        "error_message": None,
    }
    iteration.update(iteration_overrides)
    return [baseline, iteration]


class TestTraceViewContentVariants:
    def test_regression_trace_renders_rmse_not_auc(self, client, tmp_path):
        trace = _make_view_trace({})
        trace[0]["task_type"] = "regression"
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"RMSE" in response.content
        # "AUC" must not appear as a metric label (check the metric row label)
        assert b"AUC:</strong>" not in response.content

    def test_discarded_decision_renders_word_discarded(self, client, tmp_path):
        trace = _make_view_trace({"decision": "discarded", "auc_delta": -0.01, "auc_after": 0.69})
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"discarded" in response.content

    def test_error_decision_renders_word_error(self, client, tmp_path):
        trace = _make_view_trace({"decision": "error", "status": "failed", "error_message": "exec failed"})
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"error" in response.content

    def test_error_message_renders_in_html(self, client, tmp_path):
        trace = _make_view_trace({
            "decision": "error",
            "status": "failed",
            "error_message": "sandbox blocked import os",
        })
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"sandbox blocked import os" in response.content

    def test_empty_top3_summary_does_not_render_shap_box(self, client, tmp_path):
        trace = _make_view_trace({
            "shap_summary": {
                "ranked_features": [],
                "top_3_summary": "",
            },
        })
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace))
        with patch("api.main.TRACE_PATH", trace_file):
            response = client.get("/trace/view")
        assert b"SHAP" not in response.content
