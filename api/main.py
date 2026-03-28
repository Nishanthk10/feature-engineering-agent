import json
import pathlib
import tempfile
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

from agent.loop import AgentLoop

app = FastAPI(title="Feature Engineering Agent")

UPLOAD_PATH = pathlib.Path(tempfile.gettempdir()) / "fe_sandbox" / "upload.csv"
TRACE_PATH = pathlib.Path("outputs") / "trace.json"

# Global agent state — updated by the background thread
_state: dict = {"status": "idle", "iteration": 0}
_state_lock = threading.Lock()


def _run_agent(dataset_path: str, target_col: str, max_iter: int) -> None:
    with _state_lock:
        _state["status"] = "running"
        _state["iteration"] = 0

    try:
        loop = AgentLoop()

        # Monkey-patch the loop's _write_trace to track iteration count
        original_write = loop._write_trace if hasattr(loop, "_write_trace") else None

        def _patched_write(entries):
            with _state_lock:
                # entries[0] is baseline; iteration count = len - 1
                _state["iteration"] = max(0, len(entries) - 1)
            if original_write:
                original_write(entries)

        # Use module-level _write_trace via import to track progress
        import agent.loop as loop_module
        original_module_write = loop_module._write_trace

        def _tracking_write(entries):
            with _state_lock:
                _state["iteration"] = max(0, len(entries) - 1)
            original_module_write(entries)

        loop_module._write_trace = _tracking_write
        try:
            loop.run(
                dataset_path=dataset_path,
                target_col=target_col,
                max_iter=max_iter,
            )
        finally:
            loop_module._write_trace = original_module_write

    except Exception:
        pass
    finally:
        with _state_lock:
            _state["status"] = "complete"


@app.post("/run")
async def run_agent(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    max_iter: int = Form(5),
):
    UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    UPLOAD_PATH.write_bytes(contents)

    thread = threading.Thread(
        target=_run_agent,
        args=(str(UPLOAD_PATH), target_col, max_iter),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "message": "Agent running"}


@app.get("/status")
def get_status():
    with _state_lock:
        return dict(_state)


@app.get("/trace")
def get_trace():
    if not TRACE_PATH.exists():
        return {"trace": []}
    try:
        data = json.loads(TRACE_PATH.read_text())
        return {"trace": data}
    except (json.JSONDecodeError, OSError):
        return {"trace": []}


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
