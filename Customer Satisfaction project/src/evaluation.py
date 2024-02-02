import logging 
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score , mean_squared_error
class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(sel, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Retutns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(sel, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            # mse = np.average((y_true - y_pred) ** 2)
            logging.info("MSE:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in claculate MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation Strategy that uses coefficient of determination (R2)
    """
    def calculate_scores(sel, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e :
            logging.error("Error in calculating R2 Score: {}".format(e))

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_scores(sel, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE Score")
            rmse = mean_squared_error(y_pred, y_true, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e