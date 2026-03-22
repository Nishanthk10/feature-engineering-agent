import copy
import pathlib

import pandas as pd


class DatasetLoader:
    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
    MIN_ROWS = 100

    def load(self, csv_path: str, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        path = pathlib.Path(csv_path)

        if not path.exists():
            raise ValueError(f"File not found: {csv_path}")

        if path.stat().st_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File exceeds 500 MB limit: {csv_path}")

        df = pd.read_csv(path)

        if len(df) < self.MIN_ROWS:
            raise ValueError(f"Dataset has fewer than {self.MIN_ROWS} rows: {len(df)}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

        return copy.deepcopy(df), copy.deepcopy(df)
