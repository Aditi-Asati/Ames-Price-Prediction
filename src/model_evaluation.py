from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.base import RegressorMixin
import logging


class RegressionModelEvaluationStrategy:
    """
    
    """

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Performing prediction using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"MAE": mae, "MSE": mse, "R2": r2}
        logging.info(f"Model Evaluation metrics: {metrics}")
        return metrics
