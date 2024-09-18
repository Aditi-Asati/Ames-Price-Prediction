import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import mlflow

@click.command()
def main():
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    run = ml_pipeline()

if __name__=="__main__":
    main()