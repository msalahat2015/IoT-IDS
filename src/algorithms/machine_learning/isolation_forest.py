import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore


class IsolationForestClass(BaseAlgorithm):
    """
    An Isolation Forest anomaly detector.
    """
    def __init__(self, params=None):
        """
        Initializes the anomaly detector.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        self.model = IsolationForest(**self.params)

    def train(self, X, y=None):
        """
        Trains the model. Isolation Forest is unsupervised, so y is ignored.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array, ignored).
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Makes predictions using the Isolation Forest.

        :param X: Test features (NumPy array).
        :return: Raw anomaly predictions (-1 for anomaly, 1 for normal).
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance using accuracy, classification report, and ROC AUC.
        Assumes class `0` is "normal" and others are "anomalies".

        :param X: Test features (NumPy array).
        :param y: Ground truth labels (NumPy array, possibly multi-class).
        :return: Dictionary with evaluation metrics.
        """

        # Convert multiclass y to binary: 0 (normal), 1 (anomaly)
        # Modify this line to change what is considered "normal"
        y_binary = np.where(y == 0, 0, 1)

        # Predict using IsolationForest: -1 = anomaly, 1 = normal
        y_pred = self.model.predict(X)
        y_pred_binary = np.where(y_pred == -1, 1, 0)

        accuracy = accuracy_score(y_binary, y_pred_binary)
        report = classification_report(y_binary, y_pred_binary, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_binary, y_pred_binary)
        except ValueError:
            auc_roc = None  # In case of only one class in y or prediction

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "auc_roc": auc_roc
        }

    def get_model(self):
        """
        Returns the trained model.

        :return: Trained Isolation Forest model.
        """
        return self.model
