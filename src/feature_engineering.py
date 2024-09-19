from abc import ABC, abstractmethod
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    """
    
    """
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        pass


class LogTransformationStrategy(FeatureEngineeringStrategy):
    """
    
    """
    def __init__(self, features: list[str]) -> None:
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation on features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])

        logging.info("Log transformation completed.")
        return df_transformed
    
class StandardScalingStrategy(FeatureEngineeringStrategy):
    """
    
    """
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaler transformation on features : {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaler transformation completed.")
        return df_transformed
    

class MinMaxScalingStrategy(FeatureEngineeringStrategy):
    """
    
    """
    def __init__(self, features: list[str], feature_range = (0,1)) -> None:
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying min mix scaling to features : {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min max transformation completed.")
        return df_transformed
    

class OneHotEncodingStrategy(FeatureEngineeringStrategy):
    """
    
    """
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applying one got encoding on features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features
                                                       ))
        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed
    

class FeatureEngineer:
    """
    
    """
    def __init__(self, strategy: FeatureEngineeringStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)
    
