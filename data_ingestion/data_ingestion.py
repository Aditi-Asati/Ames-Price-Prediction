from abc import ABC, abstractmethod
from zipfile import ZipFile
import os
from pathlib import Path

import pandas as pd
import numpy as np


class DataExtraction(ABC):
    """
    Interface for extracting data
    """
    @abstractmethod
    def extract(self, file_path: str) -> pd.DataFrame:
        pass


class ZipDataExtraction(DataExtraction):
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

        if len(csv_files) > 1:
            raise ValueError("More than one csv file present!!")
        elif len(csv_files) == 1:
            csv_file_path = extracted_folder_path / csv_files[0]
            df = pd.read_csv(csv_file_path)
            return df
        

class DataExtractor:
    """
    """

    def extract_data(self, type: str, file_path: str):
        if type == "zip":
            return ZipDataExtraction().extract(file_path)
        else:
            raise ValueError("Invalid type")
        