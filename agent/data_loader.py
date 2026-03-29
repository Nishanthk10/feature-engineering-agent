import copy
import pathlib

import pandas as pd

from tools.schemas import TaskType


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

    def detect_task_type(
        self,
        csv_path: str,
        target_column: str,
        task_type: TaskType | None = None,
    ) -> TaskType:
        """Auto-detect or return the explicit task type for the given dataset."""
        if task_type is not None:
            return task_type

        path = pathlib.Path(csv_path)
        df = pd.read_csv(path, usecols=[target_column])
        target_series = df[target_column]
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        n_unique = target_series.nunique()
        return TaskType.regression if (is_numeric and n_unique > 20) else TaskType.classification
