import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitter:
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_variable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, self.test_size, self.random_state)
        logging.info("Train test split completed.")
        return X_train, X_test, y_train, y_test