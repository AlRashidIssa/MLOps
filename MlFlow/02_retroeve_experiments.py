import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    # Promt the user to enter a experiment_id
    # id_input = input("Enter a id for experiment!")
    # Retrieve the mlflow experiment
    experiment = get_mlflow_experiment(experiment_name="Default")


    print("Name: {}".format(experiment.name))
    print("Experoment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Creation timestamp: {}".format(experiment.creation_time))