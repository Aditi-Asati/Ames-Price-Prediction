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
    name="ames_price_predictor",
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

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        logging.info("Building and training Linear Regression model")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_columns])
        expected_columns = numerical_columns.to_list() + list(onehot_encoder.get_feature_names_out(categorical_columns))

        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error occured during model training: {e}")
        raise e
    
    finally:
        mlflow.end_run()
    
    return pipeline
    