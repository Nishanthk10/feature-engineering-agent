"""
Sandbox runner — executed as a subprocess only, never imported by the main process.

Protocol (stdin/stdout):
  stdin:  line 1 = base64-encoded pickled input DataFrame
          line 2 = base64-encoded UTF-8 transformation code
  stdout: base64-encoded pickled result dict {"df": ..., "new_columns": [...]}
  stderr: error message on failure
  exit 0: success
  exit 1: any error
"""
import ast
import base64
import pickle
import sys

ALLOWED_TOP_MODULES = {"pandas", "numpy", "scipy", "sklearn", "math", "datetime"}


def check_imports(code: str) -> None:
    """Raise ImportError if code contains any disallowed import."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
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
    try:
        lines = sys.stdin.read().splitlines()
        if len(lines) < 2:
            print("Expected 2 lines on stdin (encoded_df, encoded_code)", file=sys.stderr)
            sys.exit(1)
        encoded_df = lines[0].strip()
        encoded_code = lines[1].strip()
    except Exception as exc:
        print(f"Failed to read stdin: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pickle.loads(base64.b64decode(encoded_df))
    except Exception as exc:
        print(f"Failed to deserialize input DataFrame: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        code = base64.b64decode(encoded_code).decode("utf-8")
    except Exception as exc:
        print(f"Failed to decode transformation code: {exc}", file=sys.stderr)
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
        output = base64.b64encode(pickle.dumps(result)).decode("ascii")
        print(output)
    except Exception as exc:
        print(f"Failed to serialize output: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
