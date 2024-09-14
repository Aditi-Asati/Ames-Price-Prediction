from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class BivariateAnalysisStrategy(ABC):
    """
    
    """

    @abstractmethod
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        
        """
        pass


class NumericalVsNumericalAnalysisStrategy(BivariateAnalysisStrategy):
    """
    
    """

    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        
        """

        print(f"Plotting scatter plot of {feature1} and {feature2}")
        plt.figure(figsize=(10,8))
        sns.scatterplot(df[feature1], df[feature2])
        plt.title(f"Scatterplot of {feature1} and {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()


class NumericalVsCategoricalAnalysisStrategy(BivariateAnalysisStrategy):
    """
    
    """
    
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        
        """
        print(f"Plotting box plot of {feature1} and {feature2}")
        plt.figure(figsize=(10,8))
        sns.boxplot(df[feature1], df[feature2])
        plt.title(f"Boxplot of {feature1} and {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()

class BivariateAnalyzer:
    """
    
    """
    def __init__(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.plot(df, feature1, feature2)