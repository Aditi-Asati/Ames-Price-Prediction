from abc import ABC, abstractmethod
from typing import Union
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for different outlier detection strategies.

    This class defines the interface for outlier detection, requiring any subclass
    to implement the `detect_outlier` method.
    """
    
    @abstractmethod
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in a DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data on which to perform outlier detection.

        Returns:
        -------
        pd.DataFrame
            A DataFrame indicating outlier values (True for outliers, False otherwise).
        """
        pass


class IQROutlierDetectionStrategy(OutlierDetectionStrategy):
    """
    A class to detect outliers using the Interquartile Range (IQR) method.
    """
    
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the DataFrame using the IQR method.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze for outliers.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of boolean values where True represents an outlier.
        """
        logging.info("Detecting outliers using IQR method.")

        # Calculate the first (Q1) and third (Q3) quartiles
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        
        # Compute the Interquartile Range (IQR)
        IQR = q3 - q1
        
        # Identify outliers where values are less than Q1 - 1.5*IQR or greater than Q3 + 1.5*IQR
        outliers = (df < q1 - 1.5 * IQR) | (df > q3 + 1.5 * IQR)
        
        logging.info("Outliers detected using the IQR method.")
        return outliers

    
class ZScoreOutlierDetectionStrategy(OutlierDetectionStrategy):
    """
    A class to detect outliers using the Z-score method.
    """
    
    def __init__(self, features: list[str] = None, threshold: Union[int, float] = 3) -> None:
        """
        Initializes the ZScoreOutlierDetectionStrategy with a threshold for outlier detection and features used to detect outliers.

        Parameters:
        ----------
        features : list[str], optional
            A list of features (columns) to check for outliers. If not provided, all numeric columns are used.
        threshold : Union[int, float], default=3
            The Z-score threshold for identifying outliers. Default is 3 (for 3 standard deviations).
        """
        self.threshold = threshold
        self.features = features

    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the DataFrame based on the Z-score method.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data for outlier detection.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of boolean values where True represents an outlier.
        """
        # Initialize a DataFrame with False values indicating no outliers
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

        # Select columns to detect outliers (specified features or all numeric columns)
        if self.features:
            logging.info(f"Detecting outliers in columns: {self.features} using the Z Score method with threshold: {self.threshold}")
            columns_to_detect = self.features
        else:
            logging.info(f"Detecting outliers in all numeric features using the Z Score method with threshold: {self.threshold}")
            columns_to_detect = list(df.select_dtypes(include="number").columns)

        # Calculate Z-scores for the selected columns
        zscores = np.abs((df[columns_to_detect] - df[columns_to_detect].mean()) / df[columns_to_detect].std())

        # Mark values as outliers where Z-score exceeds the threshold
        outliers[columns_to_detect] = zscores > self.threshold

        logging.info("Outliers detected using the Z Score method.")
        return outliers
    

class OutliersDetector:
    """
    A class to manage outlier detection and handling strategies.

    It allows switching between different outlier detection strategies and provides methods to handle and visualize outliers.
    """
    
    def __init__(self, strategy: OutlierDetectionStrategy) -> None:
        """
        Initializes the OutliersDetector with a given outlier detection strategy.

        Parameters:
        ----------
        strategy : OutlierDetectionStrategy
            The initial outlier detection strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """
        Sets a new outlier detection strategy.

        Parameters:
        ----------
        strategy : OutlierDetectionStrategy
            The new outlier detection strategy to switch to.
        """
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the DataFrame using the current strategy.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data for outlier detection.

        Returns:
        -------
        pd.DataFrame
            A DataFrame indicating outliers (True for outliers, False otherwise).
        """
        logging.info("Performing outlier detection.")
        return self._strategy.detect_outlier(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove") -> pd.DataFrame:
        """
        Handles outliers in the DataFrame by either removing or capping them.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data for outlier handling.
        method : str, default="remove"
            The method to handle outliers ("remove" to drop outliers, "cap" to cap outliers).

        Returns:
        -------
        pd.DataFrame
            The cleaned DataFrame after handling outliers.
        """
        if method == "remove":
            # Detect and remove rows with outliers
            outliers = self.detect_outliers(df)
            logging.info("Removing outliers from the dataset")
            cleaned_df = df[~(outliers).any(axis=1)]
        elif method == "cap":
            # Cap the values at the 1st and 99th percentiles
            logging.info("Capping outliers in the dataset")
            cleaned_df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=0)
        else:
            logging.warning(f"Unsupported method {method}. No outliers handled.")
            return df
        
        logging.info("Outlier handling completed.")
        logging.info(f"Shape of cleaned df is : {cleaned_df.shape}")
        return cleaned_df
    

    def visualize_outliers(self, df: pd.DataFrame, features: list[str]):
        """
        Visualizes outliers in the specified features using heatmaps and boxplots.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the data for visualization.
        features : list[str]
            The list of features (columns) to visualize outliers for.
        """
        outliers = self._strategy.detect_outlier(df)
        print("Visualizing detected outliers...")

        # Generate heatmaps for each feature's outliers
        for feature in features:
            plt.figure(figsize=(10,8))
            sns.heatmap(outliers[feature], cbar=True, cmap="viridis")
            plt.title(f"Heatmap of {feature}")
            plt.show()

        print("Visualize outliers using boxplots...")

        # Generate boxplots for each feature
        for feature in features:
            plt.figure(figsize=(10,8))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(Path(__file__).parent.parent / "extracted_data" / "AmesHousing.csv")
    print(f"Shape of original df: {df.shape}")

    # Initialize outlier detector with Z-score strategy
    outlier_detector = OutliersDetector(ZScoreOutlierDetectionStrategy(["SalePrice"], threshold=3))
    
    # Handle outliers (remove them in this case)
    cleaned_df = outlier_detector.handle_outliers(df)
    print(f"Shape of cleaned df : {cleaned_df.shape}")
    # Uncomment to print cleaned DataFrame's columns
    # print(f"Cleaned df columns: {cleaned_df.columns}")

