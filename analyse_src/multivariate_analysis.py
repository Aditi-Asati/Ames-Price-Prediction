from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MultivariateAnalysisTemplate(ABC):
    """
    Abstract template for multivariate analysis of a DataFrame.
    """
    
    def execute_analysis(self, df: pd.DataFrame):
        """
        Executes the multivariate analysis on the provided DataFrame.
        
        This method orchestrates the plotting of a correlation heatmap and 
        pairwise graphs by calling the respective methods.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        self.plot_correlation_heatmap(df)
        self.plot_pairwise_graphs(df)

    @abstractmethod
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Plots the correlation heatmap for the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        pass

    @abstractmethod
    def plot_pairwise_graphs(self, df: pd.DataFrame):
        """
        Plots pairwise graphs for the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    """
    Simple implementation of multivariate analysis.
    
    This class provides concrete implementations for plotting a correlation 
    heatmap and pairwise graphs for a DataFrame.
    """
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Plots the correlation heatmap of the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        print("Plotting correlation heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr())
        plt.title("Correlation heatmap")
        plt.show()

    def plot_pairwise_graphs(self, df: pd.DataFrame):
        """
        Plots pairwise graphs of the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        print("Plotting pairwise plots...")
        plt.figure(figsize=(10, 8))
        sns.pairplot(df.corr())  
        plt.title("Correlation heatmap") 
        plt.show()
