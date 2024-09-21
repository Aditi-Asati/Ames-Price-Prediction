from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Union

# Configure logging to display the time, log level, and message
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MissingValuesHandlingStrategy(ABC):
    """
    Abstract base class for missing values handling strategies.

    This class defines the interface for handling missing values in a DataFrame.
    Subclasses must implement the `handle` method.
    """
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing missing values.

        Returns:
        -------
        pd.DataFrame
            The DataFrame after handling missing values.
        """
        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    Strategy to drop rows or columns with missing values from the DataFrame.
    """
    
    def __init__(self, axis=0, thresh=None) -> None:
        """
        Initializes the drop strategy with the axis and threshold values.

        Parameters:
        ----------
        axis : int, optional
            The axis to drop missing values along (default is 0).
        thresh : int, optional
            The minimum number of non-NA values required to retain a row/column (default is None).
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing missing values.

        Returns:
        -------
        pd.DataFrame
            The DataFrame after dropping missing values.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and threshold={self.thresh}.")
        
        # Drop missing values from rows or columns based on the axis and threshold
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        
        logging.info("Dropped missing values.")
        return df_cleaned
    

class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    Strategy to fill missing values in the DataFrame with specified methods or constants.
    """
    
    def __init__(self, method: str = "mean", fill_value: Union[str, int, float] = None) -> None:
        """
        Initializes the fill strategy with the method and fill value.

        Parameters:
        ----------
        method : str, optional
            The method to fill missing values ('mean', 'median', 'mode', 'constant') (default is 'mean').
        fill_value : Union[str, int, float], optional
            The value to use when filling missing values with the 'constant' method (default is None).
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame based on the specified method.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing missing values.

        Returns:
        -------
        pd.DataFrame
            The DataFrame after filling missing values.
        """
        df_cleaned = df.copy()  # Create a copy to avoid modifying the original DataFrame
        
        # Handle missing values based on the specified method
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
            
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())

        elif self.method == "mode":
            # Fill each column's missing values with its mode (most frequent value)
            for column in df.columns:
                df_cleaned[column] = df_cleaned[column].fillna(df[column].mode().iloc[0])

        elif self.method == "constant":
            # Fill missing values with the specified constant value
            df_cleaned = df_cleaned.fillna(self.fill_value)

        else:
            logging.warning(f"Unknown method {self.method}. No missing values handled.")
            return df_cleaned
        
        logging.info(f"Missing values filled with method {self.method} and fill value {self.fill_value}.")
        return df_cleaned
    

class MissingValuesHandler:
    """
    A class to manage missing values handling using a specified strategy.
    """
    
    def __init__(self, strategy: MissingValuesHandlingStrategy) -> None:
        """
        Initializes the MissingValuesHandler with a specific strategy.

        Parameters:
        ----------
        strategy : MissingValuesHandlingStrategy
            The missing values handling strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        """
        Updates the missing values handling strategy.

        Parameters:
        ----------
        strategy : MissingValuesHandlingStrategy
            The new missing values handling strategy to be used.
        """
        logging.info(f"Switching missing values handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the current missing values handling strategy to the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing missing values.

        Returns:
        -------
        pd.DataFrame
            The DataFrame after handling missing values.
        """
        logging.info("Executing missing values handling strategy.")
        
        # Apply the chosen strategy's handling method
        return self._strategy.handle(df)
