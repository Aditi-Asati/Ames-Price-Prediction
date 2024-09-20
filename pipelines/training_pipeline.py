from steps.data_extraction_step import data_extraction_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from zenml import Model, pipeline, step
from pathlib import Path

@pipeline(model=Model(
        # the name uniquely identifies the model
        name="ames_price_predictor"))
def ml_pipeline():
    """
    Defines an end-to-end machine learning pipeline
    """

    # data extraction step
    raw_data = data_extraction_step(file_path=str(Path(__file__).parent.parent / "archive.zip"))

    # handle missing values step
    filled_data = handle_missing_values_step(raw_data)

    # perform feature engineering step
    engineered_data = feature_engineering_step(filled_data, strategy="log", features = ["SalePrice"])

    cleaned_df = outlier_detection_step(engineered_data, strategy = "ZScore", method = "remove", feature = "SalePrice")

    X = data_splitter_step(cleaned_df, "SalePrice")

    