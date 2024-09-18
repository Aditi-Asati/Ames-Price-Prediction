from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValuesHandlingStrategy(ABC):
    """
    
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """

        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    
    """
    def __init__(self, axis=0, thresh=None) -> None:
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        logging.info(f"Dropping missing values with axis={self.axis} and threshold={self.thresh}.")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Dropped missing values.")
        return df_cleaned
    

class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    
    """
    def __init__(self, method: str = "mean", fill_value: Union[str, int, float] = None) -> None:
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
            
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())

        elif self.method == "mode":
            for column in df.columns:
                df_cleaned[column] = df_cleaned[column].fillna(df[column].mode().iloc[0])

        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)

        else:
            logging.warning(f"Unknown method {self.method}. No missing values handled.")
            return df_cleaned
        
        logging.info(f"Missing values filled with method {self.method} and fill value {self.fill_value}.")
        return df_cleaned
    

class MissingValuesHandler:
    """
        
    """
    def __init__(self, strategy: MissingValuesHandlingStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        logging.info(f"Switching missing values handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing missing values handling strategy.")
        return self._strategy.handle(df)    