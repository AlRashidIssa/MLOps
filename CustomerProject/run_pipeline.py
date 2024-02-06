from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=
                   "/home/alrashidi/Desktop/MLOps/Customer Satisfaction project/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:/home/alrashidi/.config/zenml/local_stores/a597e233-4017-4e29-9951-c4e796bf883a/mlruns"