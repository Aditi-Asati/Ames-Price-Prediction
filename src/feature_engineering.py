from abc import ABC, abstractmethod
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Configure logging to display messages with timestamps and logging levels
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract class 
class FeatureEngineeringStrategy(ABC):
    """
    Abstract base class for feature engineering strategies.

    All feature engineering strategies must implement the apply_transformation method.
    """
    
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply a transformation on a given DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame to apply the transformation to.

        Returns:
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        pass


# Define concrete classes 
class LogTransformationStrategy(FeatureEngineeringStrategy):
    """
    Feature engineering strategy to apply logarithmic transformation to specified features.
    """
    
    def __init__(self, features: list[str]) -> None:
        """
        Initializes the LogTransformationStrategy with the features to be transformed.

        Parameters:
        ----------
        features : list of str
            The features in the DataFrame on which to apply the log transformation.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the log transformation to the specified features in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with features to be transformed.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with log-transformed features.
        """
        logging.info(f"Applying log transformation on features: {self.features}")
        
        # Create a copy of the DataFrame to avoid modifying the original data
        df_transformed = df.copy()
        
        # Apply log1p (log(1 + x)) transformation on each specified feature
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])

        logging.info("Log transformation completed.")
        return df_transformed


class StandardScalingStrategy(FeatureEngineeringStrategy):
    """
    Feature engineering strategy to apply standard scaling (Z-score normalization) to specified features.
    """
    
    def __init__(self, features: list[str]) -> None:
        """
        Initializes the StandardScalingStrategy with the features to be scaled.

        Parameters:
        ----------
        features : list of str
            The features in the DataFrame to be scaled using standard scaling.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with features to be scaled.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with standardized features.
        """
        logging.info(f"Applying standard scaler transformation on features: {self.features}")
        
        # Create a copy of the DataFrame to avoid modifying the original data
        df_transformed = df.copy()
        
        # Apply standard scaling to the specified features
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        
        logging.info("Standard scaler transformation completed.")
        return df_transformed

    

class MinMaxScalingStrategy(FeatureEngineeringStrategy):
    """
    Feature engineering strategy to apply Min-Max scaling to specified features.
    """
    
    def __init__(self, features: list[str], feature_range=(0,1)) -> None:
        """
        Initializes the MinMaxScalingStrategy with the features to be scaled and the feature range.

        Parameters:
        ----------
        features : list of str
            The features in the DataFrame to be scaled using Min-Max scaling.
        feature_range : tuple, optional
            The range to which the features will be scaled (default is (0, 1)).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with features to be scaled.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with scaled features.
        """
        logging.info(f"Applying Min-Max scaling to features: {self.features}")
        
        # Create a copy of the DataFrame to avoid modifying the original data
        df_transformed = df.copy()
        
        # Apply Min-Max scaling to the specified features
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        
        logging.info("Min-Max transformation completed.")
        return df_transformed


class OneHotEncodingStrategy(FeatureEngineeringStrategy):
    """
    Feature engineering strategy to apply one-hot encoding to specified categorical features.
    """
    
    def __init__(self, features: list[str]) -> None:
        """
        Initializes the OneHotEncodingStrategy with the features to be one-hot encoded.

        Parameters:
        ----------
        features : list of str
            The categorical features in the DataFrame to be one-hot encoded.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with features to be one-hot encoded.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with the original features replaced by their encoded versions.
        """
        logging.info(f"Applying one-hot encoding on features: {self.features}")
        
        # Create a copy of the DataFrame to avoid modifying the original data
        df_transformed = df.copy()
        
        # Perform one-hot encoding on the specified features and store the result as a DataFrame
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        
        # Drop the original categorical columns from the DataFrame
        df_transformed = df_transformed.drop(columns=self.features)
        
        # Concatenate the encoded features with the remaining original features
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context class 
class FeatureEngineer:
    """
    A class to manage the feature engineering process using a given strategy.
    """
    
    def __init__(self, strategy: FeatureEngineeringStrategy) -> None:
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        ----------
        strategy : FeatureEngineeringStrategy
            The feature engineering strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Updates the feature engineering strategy to a new strategy.

        Parameters:
        ----------
        strategy : FeatureEngineeringStrategy
            The new feature engineering strategy to be used.
        """
        self._strategy = strategy

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the current feature engineering strategy to the given DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame to which the feature engineering strategy will be applied.

        Returns:
        -------
        pd.DataFrame
            The transformed DataFrame after applying the feature engineering strategy.
        """
        logging.info("Applying feature engineering strategy.")
        
        # Apply the selected strategy's transformation to the DataFrame
        return self._strategy.apply_transformation(df)

    
