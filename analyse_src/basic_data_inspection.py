from abc import ABC, abstractmethod
import pandas as pd

class InspectionStrategy(ABC):
    """
    Abstract base class for different data inspection strategies.

    This class defines the interface for any inspection strategy that inspects a DataFrame.
    """
    
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Abstract method to be implemented by concrete inspection strategies.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to inspect.
        """
        pass


class DataTypesInspectionStrategy(InspectionStrategy):
    """
    Concrete strategy for inspecting data types and non-null counts in a DataFrame.

    This strategy prints the data types of columns and the non-null count of each column.
    """

    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null count of each column in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to inspect.
        """
        print("Datatypes and non-null count:\n")
        # df.info() provides information about the DataFrame's columns, including data types and non-null values
        print(df.info())


class SummaryStatisticsInspectionStrategy(InspectionStrategy):
    """
    Concrete strategy for inspecting summary statistics of a DataFrame.

    This strategy prints descriptive statistics for numerical columns in the DataFrame.
    """

    def inspect(self, df: pd.DataFrame) -> None:
        """
        Inspects and prints summary statistics (e.g., mean, standard deviation, min, max) for numerical columns.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to inspect.
        """
        print("Summary statistics:\n")
        # df.describe() provides summary statistics such as mean, median, min, max for numeric columns
        print(df.describe())


class PerformInspection:
    """
    Class to perform inspections on a DataFrame using a specified inspection strategy.

    This class is responsible for executing the selected inspection strategy.
    """

    def __init__(self, strategy: InspectionStrategy) -> None:
        """
        Initializes the PerformInspection class with an inspection strategy.

        Parameters:
        ----------
        strategy : InspectionStrategy
            The initial inspection strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: InspectionStrategy):
        """
        Sets a new inspection strategy to be used for future inspections.

        Parameters:
        ----------
        strategy : InspectionStrategy
            The new inspection strategy to set.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the current inspection strategy on the provided DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to inspect using the current strategy.
        """
        self._strategy.inspect(df)
