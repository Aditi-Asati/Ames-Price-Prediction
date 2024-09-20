from abc import ABC, abstractmethod
from typing import Union
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    """
    
    """
    @abstractmethod
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        pass


class IQROutlierDetectionStrategy(OutlierDetectionStrategy):
    """
    
    """
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3 - q1
        outliers = (df < q1 - 1.5*IQR) | (df > q3 + 1.5*IQR)
        logging.info("Outliers detected using the IQR method.")
        return outliers
    
class ZScoreOutlierDetectionStrategy(OutlierDetectionStrategy):
    """
    
    """
    def __init__(self, features: list[str] = None, threshold: Union[int, float] = 3) -> None:
        self.threshold = threshold
        self.features = features

    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

        if self.features:
            logging.info(f"Detecting outliers in columns: {self.features} using the Z Score method with threshold: {self.threshold}")
            columns_to_detect = self.features
        else:
            logging.info(f"Detecting outliers in all numeric features using the Z Score method with threshold: {self.threshold}")
            columns_to_detect = list(df.select_dtypes(include="number").columns)

        zscores = np.abs((df[columns_to_detect] - df[columns_to_detect].mean()) / df[columns_to_detect].std())
        outliers[columns_to_detect] = zscores > self.threshold

        logging.info("Outliers detected using the Z Score method.")
        return outliers
    

class OutliersDetector:
    """
    
    """
    def __init__(self, strategy: OutlierDetectionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Performing outlier detection.")
        return self._strategy.detect_outlier(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove") -> pd.DataFrame:
        if method == "remove":
            outliers = self.detect_outliers(df)
            logging.info("Removing outliers from the dataset")
            cleaned_df = df[~(outliers).any(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset")
            cleaned_df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=0)
        else:
            logging.warning(f"Unsupported method {method}. No outliers handled.")
            return df
        
        logging.info("Outlier handling completed.")
        logging.info(f"Shape of cleaned df is : {cleaned_df.shape}")
        return cleaned_df
    

    def visualize_outliers(self, df: pd.DataFrame, features: list[str]):
        outliers = self._strategy.detect_outlier(df)
        print("Visualizing detected outliers...")
        for feature in features:
            plt.figure(figsize=(10,8))
            sns.heatmap(outliers[feature], cbar=True, cmap="viridis")
            plt.title(f"Heatmap of {feature}")
            plt.show()

        print("Visualize outliers using boxplots...")
        for feature in features:
            plt.figure(figsize=(10,8))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()


if __name__=="__main__":
    df = pd.read_csv(Path(__file__).parent.parent / "extracted_data" / "AmesHousing.csv")
    print(f"Shape of original df: {df.shape}")
    outlier_detector = OutliersDetector(ZScoreOutlierDetectionStrategy(["SalePrice"], threshold=3))
    cleaned_df = outlier_detector.handle_outliers(df)
    print(f"Shape of cleaned df : {cleaned_df.shape}")
    # print(f"Cleaned df columns: {cleaned_df.columns}")
