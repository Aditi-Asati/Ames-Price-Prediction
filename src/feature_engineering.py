from abc import ABC, abstractmethod
import pandas as pd

class FeatureEngineeringStrategy(ABC):
    """
    
    """
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        pass


class LogTransformationStrategy