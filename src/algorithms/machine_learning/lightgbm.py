import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore

class LightGBM(BaseAlgorithm):
    """
    A LightGBM classifier.
    """
    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        # Initialize the LightGBM classifier model.  We pass the parameters
        # dictionary directly to lgb.LGBMClassifier.
        self.model = lgb.LGBMClassifier(**self.params)

    def train(self, X, y):
        """
        Trains the model.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predictions (NumPy array).
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
        :return: Evaluation metrics (dictionary).
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return {"accuracy": accuracy, "classification_report": report}
    
    def get_model(self):
        """
        Returns the trained model.
        :return: The trained model
        """
        return self.model
