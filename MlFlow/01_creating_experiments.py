import mlflow
from mlflow_utils import create_mlflow_experiment


if __name__ == "__main__":
    # Create a new mlflow experiment
    experiment_id = mlflow.create_experiment(
        name= "testing_mlflow",
        artifact_location= "testing_mlflow_artifacts",
        tags={"env": "dev", "version": "1.0.0"}
    )

    print(experiment_id)

    # experiment_id = create_mlflow_experiment(experiment_name="testing_mlflow1", 
    #                                          artifacl_location="testing_mlflow1_artifact",
    #                                           tags={"env":"dev", "version":"1.0.0"})
    # print(f"Experiment ID: {experiment_id}")