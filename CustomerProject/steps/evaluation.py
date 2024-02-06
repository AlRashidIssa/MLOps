import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from src.evaluation import MSE, RMSE, R2

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
        Annotated[float, "mse"],
        Annotated[float, "rmse"],
        Annotated[float, "r2"],
    ]:
    """
    Ecaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
       prediction = model.predict(X_test)
   
       mse_class = MSE()
       mse = mse_class.calculate_scores(y_test, prediction)
   
       rmse_class = RMSE()
       rmse = rmse_class.calculate_scores(y_test, prediction)
   
       r2_class= R2()
       r2 = r2_class.calculate_scores(y_test, prediction)
       
       mlflow.log_metric("mse", mse)
       mlflow.log_metric("rmse", rmse)
       mlflow.log_metric("r2", r2)

       # return {"MSE":mse, "RMSE":rmse, "R2":r2}
       return mse,r2,rmse
    except Exception as e:
        logging.error("Error in Evaluation: {}".format(e))
        raise e 