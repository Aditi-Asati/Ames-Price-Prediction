from src.outlier_detection import OutliersDetector, ZScoreOutlierDetectionStrategy, IQROutlierDetectionStrategy, OutlierDetectionStrategy
from zenml import step
import pandas as pd
import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def outlier_detection_step(df: pd.DataFrame, strategy: str, method: str = "cap", feature: str = None):
    if strategy == "ZScore":
        outlier_detector = OutliersDetector(ZScoreOutlierDetectionStrategy())
    elif strategy == "IQR":
        outlier_detector = OutliersDetector(IQROutlierDetectionStrategy())
    else:
        logging.error(f"Unexpected outlier detection strategy : {strategy}")
        raise ValueError(f"Unexpected outlier detection strategy : {strategy}")
    
    if feature is None: 
        df_numeric = df.select_dtypes(include="number")
        cleaned_df = outlier_detector.handle_outliers(df_numeric, method)

    elif feature in df.columns:
        cleaned_df = outlier_detector.handle_outliers(df[feature], method)
        
    else:
        logging.error(f"Column {feature} does not exist in the dataframe.")
        raise ValueError(f"Column {feature} does not exist in the dataframe.")
    
    return cleaned_df