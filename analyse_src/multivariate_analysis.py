from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MultivariateAnalysisTemplate(ABC):
    """
    
    """
    def execute_analysis(self, df: pd.DataFrame):
        self.plot_correlation_heatmap(df)
        self.plot_pairwise_graphs(df)

    @abstractmethod
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        
        """
        pass

    @abstractmethod
    def plot_pairwise_graphs(self, df: pd.DataFrame):
        """
        
        """
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    """
        
    """
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
            
        """
        print("Ploting correlation heatmap...")
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr())
        plt.title("Correlation heatmap")
        plt.show()

    def plot_pairwise_graphs(self, df: pd.DataFrame):
        """
        
        """
        print("Ploting pairwise plots...")
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr())
        plt.title("Correlation heatmap")
        plt.show()



