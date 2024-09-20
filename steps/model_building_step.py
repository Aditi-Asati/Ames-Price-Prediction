import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step, Model
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:

    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_columns = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_columns.tolist()}")
    logging.info(f"Numerical columns: {numerical_columns.tolist()}")

    numerical_transformer = 