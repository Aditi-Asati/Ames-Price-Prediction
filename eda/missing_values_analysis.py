from abc import ABC, abstractmethod
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    """
    
    """

    def execute_analysis(self, df: pd.DataFrame):
        """
        
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)


    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        
        """
        pass


    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        
        """
        pass
    

class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    """
    
    """

    def identify_missing_values(self, df: pd.DataFrame):
        """
        
        """
        print("Count of missing values for each column")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])


    def visualize_missing_values(self, df: pd.DataFrame):
        """
        
        """
        print("Visualizing missing values in the dataset...")
        plt.figure(figsize=(10,8))
        sns.heatmap(df.isna(), cbar=True, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

