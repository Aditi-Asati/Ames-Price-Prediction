from abc import ABC, abstractmethod
import pandas as pd

class MissingValuesHandlingStrategy(ABC):
    """
    
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """

        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    
    """
    def __init__(self, axis=0, thresh=None) -> None:
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        return 