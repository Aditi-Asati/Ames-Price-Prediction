import os
from pathlib import Path
from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

requirements_file = Path(__file__).parent.parent / "requirements.txt"


@pipeline
def continuous_deployment_pipeline():
    trained_pipeline = ml_pipeline()

    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_pipeline)


@pipeline(enable_cache=False)
def inference_pipeline():
    batch_data = dynamic_importer()

    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step"
        )

    predictor(service=model_deployment_service, input_data=batch_data)
