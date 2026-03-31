import base64
import pickle
import subprocess
import sys
import pathlib

import pandas as pd

from agent.logger import get_logger
from tools.schemas import ExecuteResult

logger = get_logger("tools.execute")

SANDBOX_RUNNER = pathlib.Path(__file__).parent / "sandbox_runner.py"
TIMEOUT_SECONDS = 30


class ExecuteTool:
    def execute(self, df: pd.DataFrame, code: str) -> ExecuteResult:
        logger.debug(f"Sandbox: execute() entered ({df.shape[0]} rows, {df.shape[1]} cols)")

        # Serialize DataFrame and code as base64 and pass via stdin/stdout.
        # This avoids all temp-file I/O (and Windows Defender locking issues).
        try:
            encoded_df = base64.b64encode(pickle.dumps(df)).decode("ascii")
        except Exception as exc:
            logger.error(f"Sandbox: failed to serialize DataFrame: {exc}", exc_info=True)
            return ExecuteResult(
                success=False, new_columns=[], error_message=f"Failed to serialize input: {exc}", output_df=None
            )

        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        stdin_payload = encoded_df + "\n" + encoded_code
        logger.debug(f"Sandbox: launching subprocess, code length={len(code)}")

        try:
            proc = subprocess.run(
                [sys.executable, str(SANDBOX_RUNNER)],
                input=stdin_payload,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Sandbox timed out after {TIMEOUT_SECONDS} seconds")
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=f"Sandbox timed out after {TIMEOUT_SECONDS} seconds",
                output_df=None,
            )

        if proc.returncode != 0:
            error_msg = proc.stderr.strip() or "Sandbox exited with non-zero status"
            logger.warning(f"Sandbox failed (exit {proc.returncode}): {error_msg}")
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=error_msg,
                output_df=None,
            )

        try:
            result = pickle.loads(base64.b64decode(proc.stdout.strip()))
            output_df: pd.DataFrame = result["df"]
            new_columns: list[str] = result["new_columns"]
        except Exception as exc:
            logger.error(f"Sandbox: failed to deserialize output: {exc}", exc_info=True)
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=f"Failed to deserialize sandbox output: {exc}",
                output_df=None,
            )

        logger.debug(f"Sandbox: success, new_columns={new_columns}")
        return ExecuteResult(
            success=True,
            new_columns=new_columns,
            error_message=None,
            output_df=output_df,
        )
