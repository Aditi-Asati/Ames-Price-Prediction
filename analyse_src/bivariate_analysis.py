from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class BivariateAnalysisStrategy(ABC):
    """
    Abstract base class for bivariate analysis strategies.
    
    This class defines the interface for plotting relationships between 
    two features in a dataset. Specific analysis strategies should 
    implement the plot method.
    """

    @abstractmethod
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two features in the provided DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature1 (str): The name of the first feature.
        feature2 (str): The name of the second feature.
        """
        pass


class NumericalVsNumericalAnalysisStrategy(BivariateAnalysisStrategy):
    """
    Concrete strategy for analyzing the relationship between two numerical features.
    """

    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots a scatter plot of two numerical features.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature1 (str): The name of the first numerical feature.
        feature2 (str): The name of the second numerical feature.
        """

        print(f"Plotting scatter plot of {feature1} and {feature2}")
        plt.figure(figsize=(10, 8))
        sns.scatterplot(df[feature1], df[feature2])
        plt.title(f"Scatterplot of {feature1} and {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()


class NumericalVsCategoricalAnalysisStrategy(BivariateAnalysisStrategy):
    """
    Concrete strategy for analyzing the relationship between a numerical feature 
    and a categorical feature.
    """
    
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots a box plot of a numerical feature against a categorical feature.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature1 (str): The name of the numerical feature.
        feature2 (str): The name of the categorical feature.
        """
        print(f"Plotting box plot of {feature1} and {feature2}")
        plt.figure(figsize=(10, 8))
        sns.boxplot(df[feature1], df[feature2])
        plt.title(f"Boxplot of {feature1} and {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()

class BivariateAnalyzer:
    """
    Context class for performing bivariate analysis using a strategy.
    """
    
    def __init__(self, strategy: BivariateAnalysisStrategy) -> None:
        """
        Initializes the BivariateAnalyzer with a specific strategy.
        
        Parameters:
        strategy (BivariateAnalysisStrategy): The analysis strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new analysis strategy.
        
        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to use.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the analysis using the currently set strategy.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature1 (str): The name of the first feature.
        feature2 (str): The name of the second feature.
        """
        self._strategy.plot(df, feature1, feature2)
