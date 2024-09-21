from abc import ABC, abstractmethod
from zipfile import ZipFile
import os
from pathlib import Path
import pandas as pd

class DataExtractor(ABC):
    """
    Interface for extracting data from various file types.
    """
    
    @abstractmethod
    def extract(self, file_path: str) -> pd.DataFrame:
        """
        Extracts data from the specified file path and returns it as a DataFrame.
        
        Parameters:
        file_path (str): The path to the file from which to extract data.
        
        Returns:
        pd.DataFrame: The extracted data as a DataFrame.
        """
        pass


class ZipDataExtractor(DataExtractor):
    """
    Extracts data from a zip file.
    """
    
    def extract(self, file_path: str) -> pd.DataFrame:
        """
        Extracts a CSV file from a zip file and returns it as a DataFrame.
        
        Parameters:
        file_path (str): The path to the zip file.
        
        Returns:
        pd.DataFrame: The extracted CSV data as a DataFrame.
        """
        # Check if the provided file path ends with '.zip'
        if not file_path.endswith(".zip"):
            raise ValueError(f"Provided filepath doesn't contain a zip file")
        
        # Extract the contents of the zip file to a specified folder
        with ZipFile(file_path, "r") as zip_ref:
            extracted_folder_path = Path(__file__).parent.parent / "extracted_data"
            zip_ref.extractall(extracted_folder_path)

        # List files in the extracted directory and filter for CSV files
        files = os.listdir(extracted_folder_path)
        csv_files = [file for file in files if file.endswith(".csv")]

        # Handle cases for found CSV files
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found.")

        # Read the single CSV file into a DataFrame
        csv_file_path = extracted_folder_path / csv_files[0]
        df = pd.read_csv(csv_file_path)
        return df
        

class DataExtractionFactory:
    """
    Factory class for creating data extractors based on file extension.
    """
    
    @staticmethod
    def get_data_extractor(file_extension: str) -> DataExtractor:
        """
        Returns a data extractor for the specified file extension.
        
        Parameters:
        file_extension (str): The file extension to determine the extractor.
        
        Returns:
        DataExtractor: An instance of the appropriate data extractor.
        """
        if file_extension == "zip":
            return ZipDataExtractor()
        else:
            raise ValueError(f"No extractor available for file extension: {file_extension}")
        

if __name__ == "__main__":
    # Path to the zip file containing the data
    file_path = str(Path(__file__).parent.parent / "archive.zip")
    # Create an instance of the ZipDataExtractor
    zipdata_extractor = DataExtractionFactory.get_data_extractor("zip")
    # Extract data from the zip file into a DataFrame
    df = zipdata_extractor.extract(file_path)
    # Print the shape of the extracted DataFrame
    print(df.shape)
