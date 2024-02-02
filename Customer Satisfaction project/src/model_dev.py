import logging 
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Trainind labels
        Returns:
            None 
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Trainind labels
        Returns:
            None 

        reg = LinearRegression(**kwargs) creates a linear regression model 
        object (reg) with optional parameters specified by the kwargs dictionary. 
        This allows you to customize the behavior of the linear regression model by 
        passing different parameters when instantiating the LinearRegression object.
        For example:
        kwargs = {'fit_intercept': True, 'normalize': False}
        reg = LinearRegression(**kwargs)
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trainig completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e