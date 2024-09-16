from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    """
    
    """

    @abstractmethod
    def visualize(self, df: pd.DataFrame, feature: str):
        pass


class NumericalFeatureAnalysis(UnivariateAnalysisStrategy):
    """
    
    """

    def visualize(self, df: pd.DataFrame, feature: str):
        print(f"Visualizing feature : {feature}")
        plt.figure(figsize=(10,8))
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f"Histogram of {feature}")
        plt.show()


class CategoricalFeatureAnalysis(UnivariateAnalysisStrategy):
    """
    
    """

    def visualize(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10,8))
        sns.countplot(df[feature])
        plt.title(f"Countplot of {feature}")
        plt.show()


class UnivariateAnalysis:

    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.visualize(df, feature)