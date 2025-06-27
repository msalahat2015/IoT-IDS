import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class

class LogisticRegressionClassifier(BaseAlgorithm):
    """
    A Logistic Regression classifier.
    """
    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}  # Default parameters if none are provided
        self.params = params
        self.model = LogisticRegression(**self.params) # Initialize the LogisticRegression model

    def train(self, X, y):
        """
        Trains the model.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array).
        """
        self.model.fit(X, y)  # Train the LogisticRegression model

    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predictions (NumPy array).
        """
        return self.model.predict(X)  # Return the predictions made by the model

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
        :return: Evaluation metrics (dictionary).
        """
        y_pred = self.predict(X)  # Get predictions on the test set
        accuracy = accuracy_score(y, y_pred)  # Calculate the accuracy
        report = classification_report(y, y_pred)  # Get a detailed classification report
        return {"accuracy": accuracy, "classification_report": report}  # Return the metrics
    
    def get_model(self):
        """
        Returns the trained model.
        :return: The trained model
        """
        return self.model
