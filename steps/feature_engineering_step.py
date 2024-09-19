from src.feature_engineering import FeatureEngineer, LogTransformationStrategy, StandardScalingStrategy, MinMaxScalingStrategy, OneHotEncodingStrategy
import pandas as pd
from zenml import step

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str, features: list[str]) -> pd.DataFrame:
    if strategy == "log":
        feature_transformer = FeatureEngineer(LogTransformationStrategy(features))
    
    elif strategy == "min-max":
        feature_transformer = FeatureEngineer(MinMaxScalingStrategy(features))
    
    elif strategy == "standard_scaler":
        feature_transformer = FeatureEngineer(StandardScalingStrategy(features))
    
    elif strategy == "one-hot":
        feature_transformer = FeatureEngineer(OneHotEncodingStrategy(features))
        
    else:
        raise ValueError(f"Unsupported feature engineering strategy : {strategy}")

    df_transformed = feature_transformer.apply_transformation(df)
    return df_transformed