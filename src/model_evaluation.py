from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.base import RegressorMixin
import logging

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RegressionModelEvaluationStrategy:
    """
    A strategy class to evaluate regression models based on common metrics.

    This class provides a method to evaluate the performance of regression models using
    Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) metrics.
    """
    
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a trained regression model using test data and computes performance metrics.

        Parameters:
        ----------
        model : RegressorMixin
            The trained regression model (must implement the `predict` method).
        X_test : pd.DataFrame
            The test set features to make predictions on.
        y_test : pd.Series
            The true values of the target variable for the test set.

        Returns:
        -------
        dict
            A dictionary containing the calculated metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE),
            and R-squared (R2).
        """
        
        # Perform predictions using the provided model on the test data
        logging.info("Performing prediction using the trained model.")
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics: MAE, MSE, and R2
        logging.info("Calculating evaluation metrics.")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store the computed metrics in a dictionary and return
        metrics = {"MAE": mae, "MSE": mse, "R2": r2}
        logging.info(f"Model Evaluation metrics: {metrics}")
        return metrics
