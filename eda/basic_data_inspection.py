from abc import ABC, abstractmethod
import pandas as pd

class InspectionStrategy(ABC):
    """
    
    """
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        pass


class DataTypesInspectionStrategy(InspectionStrategy):
    """
    
    """

    def inspect(self, df: pd.DataFrame):
        print("Datatypes and non-null count:\n")
        print(df.info())

class SummaryStaticticsInspectionStrategy(InspectionStrategy):
    """
    
    """

    def inspect(self, df: pd.DataFrame) -> None:
        print("Summary statistics:\n")
        print(df.describe())



class PerformInspection:
    """
    
    """

    def __init__(self, strategy: InspectionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: InspectionStrategy):
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)