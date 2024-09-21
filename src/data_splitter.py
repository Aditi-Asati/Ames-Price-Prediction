import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logging to display messages with timestamp and level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitter:
    """
    A class used to handle train-test splitting of a dataset.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Initializes the DataSplitter with test size and random state.

        Parameters:
        ----------
        test_size : float, optional
            Proportion of the dataset to be used for testing (default is 0.2).
        random_state : int, optional
            Random seed for shuffling data before splitting (default is 42).
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_variable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the input dataframe into training and testing sets for both features and the target variable.

        Parameters:
        ----------
        df : pd.DataFrame
            The input dataset to be split.
        target_variable : str
            The column name of the target variable to be predicted.

        Returns:
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing:
            - X_train : Training set features (DataFrame)
            - X_test : Testing set features (DataFrame)
            - y_train : Training set target variable (Series)
            - y_test : Testing set target variable (Series)
        """
        # Split the dataframe into features (X) and target (y)
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Log the completion of the split and the shapes of the resulting datasets
        logging.info("Train test split completed.")
        logging.info(f"Shapes of x train test, y train test : {X_train.shape}, {X_test.shape}, {y_train.shape}")
