from abc import ABC, abstractmethod
from zipfile import ZipFile
import os
from pathlib import Path

import pandas as pd


class DataExtractor(ABC):
    """
    Interface for extracting data
    """
    @abstractmethod
    def extract(self, file_path: str) -> pd.DataFrame:
        pass


class ZipDataExtractor(DataExtractor):
    """
    Extracts data from a zipfile
    """
    def extract(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError(f"Provided filepath doesn't contain a zip file")
        
        with ZipFile(file_path, "r") as zip_ref:
            extracted_folder_path = Path(__file__).parent.parent / "extracted_data"
            zip_ref.extractall(extracted_folder_path)

        files = os.listdir(extracted_folder_path)
        csv_files = [file for file in files if file.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found.")

        
        csv_file_path = extracted_folder_path / csv_files[0]
        df = pd.read_csv(csv_file_path)
        return df
        

class DataExtractionFactory:
    """
    """
    @staticmethod
    def get_data_extractor(file_extension: str) -> DataExtractor:
        if type == "zip":
            return ZipDataExtractor()
        else:
            raise ValueError(f"No extractor available for file extension: {file_extension}")
        

if __name__=="__main__":
    file_path = str(Path(__file__).parent.parent / "archive.zip")
    # print(type(file_path))
    zipdata_extractor = DataExtractionFactory.get_data_extractor("zip")
    df = zipdata_extractor.extract(file_path)
    print(df.shape)