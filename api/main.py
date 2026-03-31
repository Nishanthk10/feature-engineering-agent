import html as _html
import json
import pathlib
import tempfile
import threading
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

load_dotenv()

from agent.logger import get_logger
from agent.loop import AgentLoop
from tools.schemas import TaskType

logger = get_logger("api")

app = FastAPI(title="Feature Engineering Agent")

UPLOAD_PATH = pathlib.Path(tempfile.gettempdir()) / "fe_sandbox" / "upload.csv"
TRACE_PATH = pathlib.Path("outputs") / "trace.json"

# Global agent state — updated by the background thread
_state: dict = {"status": "idle", "iteration": 0}
_state_lock = threading.Lock()


def _run_agent(dataset_path: str, target_col: str, max_iter: int,
               task_type: str | None = None) -> None:
    # 1. Mark running immediately — before any other operation
    with _state_lock:
        _state["status"] = "running"
        _state["iteration"] = 0

    # 2. Clear stale trace so the UI does not show results from a previous run
    try:
        if TRACE_PATH.exists():
            TRACE_PATH.unlink()
    except OSError as e:
        print(f"[_run_agent warning] could not delete stale trace: {e}")

    try:
        # 3. Confirm uploaded file exists before handing off to AgentLoop
        if not pathlib.Path(dataset_path).exists():
            print(f"[_run_agent error] uploaded file not found: {dataset_path}")
            raise FileNotFoundError(f"uploaded file not found: {dataset_path}")
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
                task_type=TaskType(task_type) if task_type is not None else None,
            )
        finally:
            loop_module._write_trace = original_module_write

    except BaseException as e:
        print(f"[_run_agent error] agent raised {type(e).__name__}: {e}")
        traceback.print_exc()
        logger.error(f"Agent run failed ({type(e).__name__}): {e}", exc_info=True)
    finally:
        with _state_lock:
            _state["status"] = "complete"


@app.post("/run")
async def run_agent(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    max_iter: int = Form(5),
    task_type: str = Form("auto"),
):
    UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    UPLOAD_PATH.write_bytes(contents)

    resolved_task_type = None if task_type == "auto" else task_type

    thread = threading.Thread(
        target=_run_agent,
        args=(str(UPLOAD_PATH), target_col, max_iter, resolved_task_type),
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
    with _state_lock:
        status = _state["status"]
    if not TRACE_PATH.exists():
        return {"trace": [], "status": status}
    try:
        data = json.loads(TRACE_PATH.read_text())
        return {"trace": data, "status": status}
    except (json.JSONDecodeError, OSError):
        return {"trace": [], "status": status}


def _render_trace_html(trace: list[dict]) -> str:
    esc = _html.escape

    baseline = next((e for e in trace if e.get("status") == "baseline"), None)
    iterations = [e for e in trace if e.get("status") != "baseline"]

    task_type = baseline.get("task_type", "classification") if baseline else "classification"
    baseline_metric = (baseline.get("primary_metric") or baseline.get("auc")) if baseline else None
    metric_label = "RMSE" if task_type == "regression" else "AUC"

    p = []
    p.append("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agent Reasoning Trace</title>
  <style>
    body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 16px;color:#222}
    h1{font-size:1.6rem;border-bottom:2px solid #333;padding-bottom:8px}
    h2{font-size:1.1rem;margin:24px 0 8px}
    .baseline{background:#f5f5f5;border-left:4px solid #555;padding:12px 16px;
              border-radius:0 4px 4px 0;margin:16px 0}
    .iter-card{border:1px solid #ddd;border-radius:6px;padding:14px 18px;margin:14px 0}
    .iter-header{font-size:1rem;font-weight:bold;margin-bottom:10px}
    blockquote{border-left:3px solid #aaa;margin:6px 0;padding:4px 12px;
               color:#444;font-style:italic}
    pre{background:#f0f0f0;padding:10px 14px;border-radius:4px;
        overflow-x:auto;font-size:.88em;margin:6px 0}
    .metric-row{font-family:monospace;margin:6px 0}
    .kept{color:#1a7a1a;font-weight:bold}
    .discarded{color:#666;font-weight:bold}
    .error{color:#b00020;font-weight:bold}
    .shap-box{background:#fafafa;border:1px solid #e0e0e0;border-radius:4px;
              padding:8px 12px;margin:6px 0;font-size:.9em}
    .summary{background:#e8f4e8;border:1px solid #99cc99;padding:14px 18px;
             border-radius:6px;margin-top:28px}
    .no-trace{color:#888;font-style:italic}
  </style>
</head>
<body>
<h1>Agent Reasoning Trace</h1>""")

    if not trace:
        p.append('<p class="no-trace">No trace available yet.</p>')
    else:
        # Baseline section
        p.append("<h2>Baseline</h2>")
        if baseline:
            bm = f"{baseline_metric:.4f}" if baseline_metric is not None else "—"
            feats = ", ".join(esc(f) for f in baseline.get("features_used", []))
            p.append(
                f'<div class="baseline">'
                f"<strong>Task type:</strong> {esc(task_type)}&nbsp;&nbsp;"
                f"<strong>Baseline {esc(metric_label)}:</strong> {esc(bm)}"
                + (f"&nbsp;&nbsp;<strong>Features:</strong> {feats}" if feats else "")
                + "</div>"
            )

        # Iteration cards
        if iterations:
            p.append("<h2>Iterations</h2>")

        for rec in iterations:
            decision = rec.get("decision", "")
            dec_class = "kept" if decision == "kept" else ("error" if decision == "error" else "discarded")
            ab = rec.get("auc_before", 0.0)
            aa = rec.get("auc_after", 0.0)
            ad = rec.get("auc_delta", 0.0)
            delta_str = f"+{ad:.4f}" if ad >= 0 else f"{ad:.4f}"

            shap = rec.get("shap_summary") or {}
            top3 = esc(shap.get("top_3_summary", ""))
            err = rec.get("error_message")

            p.append('<div class="iter-card">')
            p.append(
                f'<div class="iter-header">'
                f"Iteration {esc(str(rec.get('iteration', '?')))} "
                f"&mdash; {esc(rec.get('status', ''))}"
                f"</div>"
            )
            p.append(f"<strong>Hypothesis:</strong><blockquote>{esc(rec.get('hypothesis', ''))}</blockquote>")
            p.append(f"<strong>Feature:</strong> <code>{esc(rec.get('feature_name', ''))}</code>")
            p.append(f"<pre>{esc(rec.get('transformation_code', ''))}</pre>")
            p.append(
                f'<div class="metric-row">'
                f"<strong>{esc(metric_label)}:</strong> "
                f"{ab:.4f} &rarr; {aa:.4f} (delta: {esc(delta_str)})"
                f"</div>"
            )
            p.append(f"<div><strong>Decision:</strong> <span class=\"{dec_class}\">{esc(decision)}</span></div>")
            if top3:
                p.append(f'<div class="shap-box"><strong>SHAP top&nbsp;3:</strong> {top3}</div>')
            if err:
                p.append(
                    f'<div style="color:#b00020;font-size:.9em;margin-top:6px">'
                    f"<strong>Error:</strong> {esc(err)}</div>"
                )
            p.append("</div>")

        # Final summary
        kept = [r for r in iterations if r.get("decision") == "kept"]
        last_kept = kept[-1] if kept else None
        final_metric = last_kept["auc_after"] if last_kept else baseline_metric
        if baseline_metric is not None and final_metric is not None:
            lift_str = f"{final_metric - baseline_metric:+.4f}"
        else:
            lift_str = "—"
        kept_names = ", ".join(esc(r.get("feature_name", "")) for r in kept) or "none"

        p.append(
            f'<div class="summary">'
            f"<strong>Summary</strong><br>"
            f"Task type: {esc(task_type)}<br>"
            f"Lift achieved ({esc(metric_label)}): {esc(lift_str)}<br>"
            f"Features kept: {kept_names}"
            f"</div>"
        )

    p.append("</body></html>")
    return "\n".join(p)


@app.get("/trace/view")
def view_trace():
    if not TRACE_PATH.exists():
        return HTMLResponse(_render_trace_html([]))
    try:
        data = json.loads(TRACE_PATH.read_text())
        return HTMLResponse(_render_trace_html(data))
    except (json.JSONDecodeError, OSError):
        return HTMLResponse(_render_trace_html([]))


@app.get("/logs")
def get_logs():
    log_path = pathlib.Path("outputs/agent.log")
    if not log_path.exists():
        return PlainTextResponse("No logs yet.")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return PlainTextResponse("\n".join(lines[-100:]))


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
