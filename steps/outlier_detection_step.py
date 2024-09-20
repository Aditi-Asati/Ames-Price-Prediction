from src.outlier_detection import OutliersDetector, ZScoreOutlierDetectionStrategy, IQROutlierDetectionStrategy, OutlierDetectionStrategy
from zenml import step
import pandas as pd
import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def outlier_detection_step(df: pd.DataFrame, strategy: str, method: str = "remove", features: list[str] = None):
    if strategy == "ZScore":
        outlier_detector = OutliersDetector(ZScoreOutlierDetectionStrategy(features=features))
    elif strategy == "IQR":
        outlier_detector = OutliersDetector(IQROutlierDetectionStrategy())
    else:
        logging.error(f"Unexpected outlier detection strategy : {strategy}")
        raise ValueError(f"Unexpected outlier detection strategy : {strategy}")

    if all([feature in df.columns for feature in features]):
        cleaned_df = outlier_detector.handle_outliers(df=df, method=method)

    else:
        logging.error(f"Column {features} does not exist in the dataframe.")
        raise ValueError(f"Column {features} does not exist in the dataframe.")
    
    logging.info(f"Shape of cleaned df is : {cleaned_df.shape}")
    return cleaned_df