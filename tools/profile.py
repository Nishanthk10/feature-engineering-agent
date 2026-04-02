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

        sample = df.head(5).to_dict(orient="records")
        sample_rows = [
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in row.items()
             if k != target_col}
            for row in sample
        ]

        return DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            target_col=target_col,
            feature_cols=feature_cols,
            missing_rate=missing_rate,
            dtypes=dtypes,
            sample_rows=sample_rows,
        )
