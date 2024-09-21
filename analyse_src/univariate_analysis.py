from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    """
    Abstract base class for univariate analysis strategies.
    """

    @abstractmethod
    def visualize(self, df: pd.DataFrame, feature: str):
        """
        Visualizes a specified feature from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature to visualize.
        """
        pass


class NumericalFeatureAnalysis(UnivariateAnalysisStrategy):
    """
    Concrete implementation for analyzing numerical features.
    
    This class provides a method to visualize numerical features using 
    histograms.
    """

    def visualize(self, df: pd.DataFrame, feature: str):
        """
        Visualizes a numerical feature using a histogram with a KDE overlay.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature (str): The name of the numerical feature to visualize.
        """
        print(f"Visualizing feature: {feature}")
        plt.figure(figsize=(10, 8))
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f"Histogram of {feature}")
        plt.show()


class CategoricalFeatureAnalysis(UnivariateAnalysisStrategy):
    """
    Concrete implementation for analyzing categorical features.
    
    This class provides a method to visualize categorical features using 
    count plots.
    """

    def visualize(self, df: pd.DataFrame, feature: str):
        """
        Visualizes a categorical feature using a count plot.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature (str): The name of the categorical feature to visualize.
        """
        plt.figure(figsize=(10, 8))
        sns.countplot(df[feature])
        plt.title(f"Countplot of {feature}")
        plt.show()


class UnivariateAnalysis:
    """
    Context class for performing univariate analysis using a strategy.
    """

    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalysis with a specific strategy.
        
        Parameters:
        strategy (UnivariateAnalysisStrategy): The analysis strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new analysis strategy.
        
        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to use.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the analysis using the currently set strategy.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature to visualize.
        """
        self._strategy.visualize(df, feature)
