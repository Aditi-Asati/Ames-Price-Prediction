from abc import ABC, abstractmethod
from typing import Union
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, threshold: Union[int, float] = 3) -> None:
        self.threshold = threshold


    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Detecting outliers using the Z Score method with threshold: {self.threshold}")
        zscores = np.abs((df - df.mean())/ df.std())
        outliers = pd.DataFrame(zscores > self.threshold, columns=df.columns)
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
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset")
            cleaned_df = df[~(outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset")
            cleaned_df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=0)
        else:
            logging.warning(f"Unsupported method {method}. No outliers handled.")
            return df
        
        logging.info("Outlier handling completed.")
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


