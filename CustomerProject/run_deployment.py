import click 

from pipelines.deployment_pipeline import deployment_pipeline, inference_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_Deployer import MLFlowModelDeployer
from zenm.integrations.mlflow.services import MLFlowDeploymentService
# Define choice constants
DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help=("Optionally choose to only run the deployment pipeline to train and deploy a model "
          "('deploy'), or to only run a prediction against the deployed model ('predict'). "
          "By default both will be run ('deploy_and_predict')."),
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)
def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = 
    if config == DEPLOY or config == DEPLOY_AND_PREDICT:
        deployment_pipeline(min_accuracy)
    if config == PREDICT or config == DEPLOY_AND_PREDICT:
        inference_pipeline()

if __name__ == "__main__":
    run_deployment()
