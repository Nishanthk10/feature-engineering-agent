"""
Sandbox runner — executed as a subprocess only, never imported by the main process.

Usage: python sandbox_runner.py <base64-encoded-code>
Exit 0: success, writes output.pkl
Exit 1: any error, writes error message to stderr
"""
import ast
import base64
import pickle
import sys
import pathlib
import tempfile

SANDBOX_DIR = pathlib.Path(tempfile.gettempdir()) / "fe_sandbox"

ALLOWED_TOP_MODULES = {"pandas", "numpy", "scipy", "sklearn", "math", "datetime"}


def check_imports(code: str) -> None:
    """Raise ImportError if code contains any disallowed import."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # SyntaxError will surface again on exec(); let it pass through here.
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_TOP_MODULES:
                    raise ImportError(f"Disallowed import: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_TOP_MODULES:
                    raise ImportError(f"Disallowed import from: '{node.module}'")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sandbox_runner.py <base64-encoded-code>", file=sys.stderr)
        sys.exit(1)

    try:
        code = base64.b64decode(sys.argv[1]).decode("utf-8")
    except Exception as exc:
        print(f"Failed to decode code argument: {exc}", file=sys.stderr)
        sys.exit(1)

    input_path = SANDBOX_DIR / "input.pkl"
    output_path = SANDBOX_DIR / "output.pkl"

    try:
        with open(input_path, "rb") as fh:
            df = pickle.load(fh)
    except Exception as exc:
        print(f"Failed to load input.pkl: {exc}", file=sys.stderr)
        sys.exit(1)

    original_cols = set(df.columns)

    try:
        check_imports(code)
        exec(code, {"df": df})  # noqa: S102 — intentional, sandboxed subprocess only
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    new_cols = [c for c in df.columns if c not in original_cols]

    try:
        result = {"df": df, "new_columns": new_cols}
        with open(output_path, "wb") as fh:
            pickle.dump(result, fh)
    except Exception as exc:
        print(f"Failed to write output.pkl: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
