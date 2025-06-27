import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore

class AdaBoost(BaseAlgorithm):
    """
    An AdaBoost classifier.
    """
    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        # Initialize the AdaBoost classifier model.
        self.model = AdaBoostClassifier(**self.params)

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
