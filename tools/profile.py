import pandas as pd

from tools.schemas import DatasetProfile


class ProfileTool:
    def profile(self, df: pd.DataFrame, target_col: str) -> DatasetProfile:
        feature_cols = [c for c in df.columns if c != target_col]

        missing_rate = {
            col: float(df[col].isna().sum() / len(df))
            for col in feature_cols
        }

        dtypes = {col: str(df[col].dtype) for col in df.columns}

        return DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            target_col=target_col,
            feature_cols=feature_cols,
            missing_rate=missing_rate,
            dtypes=dtypes,
        )
