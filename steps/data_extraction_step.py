import pandas as pd
from data_ingestion.data_ingestion import DataExtractionFactory
from zenml import step


@step
def data_extraction_step(file_path: str) -> pd.DataFrame:

    file_extension = file_path.split(".")[-1]
    data_extractor = DataExtractionFactory().get_data_extractor(file_extension)
    df = data_extractor.extract(file_path)
    return df
