import click
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="stop the prediction service when done"
)
def run_main(stop_service: bool):
    model_name = "ames_price_predictor"
    if stop_service:
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return
    
    continuous_deployment_pipeline()

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    inference_pipeline()

    print()

    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
    )

    if service[0]:
        print()


if __name__=="__main__":
    run_main()

