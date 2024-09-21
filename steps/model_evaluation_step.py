from src.model_evaluation import RegressionModelEvaluationStrategy
import pandas as pd
from zenml import step
from sklearn.pipeline import Pipeline
import logging

@step
def model_evaluation_step(trained_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):

    logging.info("Applying the same preprocessing to the test data.")

    X_test_processed = trained_pipeline.named_steps["preprocessor"].transform(X_test)

    evaluator = RegressionModelEvaluationStrategy()
    trained_model = trained_pipeline.named_steps["model"]
    metrics = evaluator.evaluate(trained_model, X_test_processed, y_test)
    mse = metrics["MSE"]
    return metrics, mse