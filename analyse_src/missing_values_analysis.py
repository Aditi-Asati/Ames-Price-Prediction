from abc import ABC, abstractmethod
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class MissingValuesAnalysisTemplate(ABC):
    """
    Abstract template for analyzing missing values in a DataFrame.
    """

    def execute_analysis(self, df: pd.DataFrame):
        """
        Executes the analysis on the provided DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes the missing values in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to visualize.
        """
        pass
    

class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    """
    Simple implementation of missing values analysis.
    
    This class provides concrete implementations for identifying and 
    visualizing missing values in a DataFrame.
    """

    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies and prints the count of missing values for each column.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        print("Count of missing values for each column")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes the missing values in the DataFrame using a heatmap.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to visualize.
        """
        print("Visualizing missing values in the dataset...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.isna(), cbar=True, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
