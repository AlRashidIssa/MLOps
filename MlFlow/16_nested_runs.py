import mlflow
from mlflow_utils import create_mlflow_experiment

experiment_id = create_mlflow_experiment(
    experiment_name= "Nested Runs",
    artifact_location= "nested_run_artifacts",
    tags={"purpose":"learning"}
)


with mlflow.start_run(run_name="Image Classification") as IMC:
    print("RUN ID Image Classification:", IMC.info.run_id)

    mlflow.log_param("loss_error_IMC", 0.025)

    with mlflow.start_run(run_name="Binarry Classification",nested=True) as BIC:
        print("RUN ID Binary CLassification:", BIC.info.run_id)
        mlflow.log_param("loss_error_BIC", 0.8)

        with mlflow.start_run(run_name="Over Fiting", nested=True) as OVF:
            print("RUN ID OverFiting:", OVF.info.run_id )
            mlflow.log_param("HIGH_LOW_OVF", "HIGH")

        with mlflow.start_run(run_name="Classification Images Cats/Dogs", nested=True) as CICD:
            print("RUN ID Classification Images Cats/Dogs:", CICD.info.run_id)
            mlflow.log_param("loss_errer_CICD", 0.0003)

    with mlflow.start_run(run_name="Multi Classification", nested=True) as MUC:
        print("RUN ID Multi CLassification:", MUC.info.run_id)
        mlflow.log_param("loss_error_MUC", 0.09)