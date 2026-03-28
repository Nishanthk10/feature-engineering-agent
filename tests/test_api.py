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
