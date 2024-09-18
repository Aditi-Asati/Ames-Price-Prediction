from src.handle_missing_values import MissingValuesHandler, FillMissingValuesStrategy, DropMissingValuesStrategy
import pandas as pd
from typing import Any
from zenml import step


@step
def handle_missing_values_step(df: pd.DataFrame, method: str = "mean", fill_value: Any = None) -> pd.DataFrame:
    if method == "drop":
        handler = MissingValuesHandler(DropMissingValuesStrategy())
    
    elif method in ["mean", "median", "mode", "constant"]:
        handler = MissingValuesHandler(FillMissingValuesStrategy(method, fill_value))
    
    else:
        raise ValueError(f"Unsupported missing values handling strategy: {method}")
    
    df_cleaned = handler.handle_missing_values(df)
    return df_cleaned