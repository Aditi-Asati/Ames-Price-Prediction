from zenml import step
import pandas as pd
from src.data_splitter import DataSplitter

@step()
def data_splitter_step(df: pd.DataFrame, target_variable: str, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_splitter = DataSplitter(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = data_splitter.split_data(df, target_variable)
    return X_train, X_test, y_train, y_test