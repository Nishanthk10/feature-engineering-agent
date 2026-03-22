import base64
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile

import pandas as pd

from tools.schemas import ExecuteResult

SANDBOX_DIR = pathlib.Path(tempfile.gettempdir()) / "fe_sandbox"
SANDBOX_RUNNER = pathlib.Path(__file__).parent / "sandbox_runner.py"
TIMEOUT_SECONDS = 30


class ExecuteTool:
    def execute(self, df: pd.DataFrame, code: str) -> ExecuteResult:
        os.makedirs(SANDBOX_DIR, exist_ok=True)

        input_path = SANDBOX_DIR / "input.pkl"
        output_path = SANDBOX_DIR / "output.pkl"

        with open(input_path, "wb") as fh:
            pickle.dump(df, fh)

        # Encode code as base64 to avoid shell-escaping issues on all platforms.
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")

        try:
            proc = subprocess.run(
                [sys.executable, str(SANDBOX_RUNNER), encoded_code],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=f"Sandbox timed out after {TIMEOUT_SECONDS} seconds",
                output_df=None,
            )

        if proc.returncode != 0:
            error_msg = proc.stderr.strip() or "Sandbox exited with non-zero status"
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=error_msg,
                output_df=None,
            )

        try:
            with open(output_path, "rb") as fh:
                result = pickle.load(fh)
            output_df: pd.DataFrame = result["df"]
            new_columns: list[str] = result["new_columns"]
        except Exception as exc:
            return ExecuteResult(
                success=False,
                new_columns=[],
                error_message=f"Failed to read sandbox output: {exc}",
                output_df=None,
            )

        return ExecuteResult(
            success=True,
            new_columns=new_columns,
            error_message=None,
            output_df=output_df,
        )
