from mlflow_utils import create_mlflow_experiment
from mlflow import MlflowClient

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="model_registry",
        artifact_location="model_registry_artifacts",
        tags={"purpose":"learning"}    
        )
    
    print("Experiment ID:",experiment_id)

    client = MlflowClient()
    model_name = "registered_model_1"

    # create registered model
    client.create_registered_model(name=model_name)

    # create model version
    source = "/home/alrashidi/Desktop/MLOps/MlFlow/model_registry_artifacts/07e343ceb7d04bf4aadcd1123f0b0ea2/artifacts/rft_model2"
    run_id = "07e343ceb7d04bf4aadcd1123f0b0ea2"
    client.create_model_version(name=model_name, source=source, run_id=run_id)

    # transition model version stage
    client.transition_model_version_stage(name=model_name, version=1, stage="Archived")

    # delete model version
    client.delete_model_version(name=model_name, version=1)

    # delete registered model
    client.delete_registered_model(name=model_name)

    # adding descriprion to registired mode.
    client.update_registered_model(name=model_name, description="This is a test model")

    # adding tags to registired model.
    client.set_registered_model_tag(name=model_name, key="tag1", value="value1")

    # adding description to model version.
    client.update_model_version(name=model_name, version=1, description="This is a test model version")

    # adding tags to model version.
    client.set_model_version_tag(name=model_name, version=1, key="tag1", value="value1")