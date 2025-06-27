import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore

class LOFClass(BaseAlgorithm):
    """
    A Local Outlier Factor anomaly detector.
    LOF is unsupervised and detects anomalies (outliers) in the dataset.
    """

    def __init__(self, params=None):
        """
        Initializes the LOF model with given parameters.

        :param params: Dictionary of parameters for LocalOutlierFactor.
        """
        if params is None:
            params = {}
        self.params = params
        # Use novelty=True to allow prediction on unseen data
        self.model = LocalOutlierFactor(novelty=True, **params)

    def train(self, X, y=None):
        """
        Fits the LOF model on training data.

        :param X: Training features (NumPy array or DataFrame).
        :param y: Ignored (LOF is unsupervised).
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Predicts outliers on the test set.

        :param X: Test features.
        :return: Array with values: -1 for outliers, 1 for inliers.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model. Assumes y contains 0 for benign, and other integers (e.g., 1, 2, 3) for attacks.

        :param X: Test features.
        :param y: True labels (0 = Benign, others = Anomalies).
        :return: Dictionary with evaluation metrics.
        """
        # Predict: -1 = anomaly, 1 = normal
        y_pred = self.predict(X)

        # Convert to binary: 1 = anomaly, 0 = benign
        y_pred_binary = np.where(y_pred == -1, 1, 0)

        # Convert y to binary for consistent comparison
        y_binary = np.where(y == 0, 0, 1)

        accuracy = accuracy_score(y_binary, y_pred_binary)
        report = classification_report(y_binary, y_pred_binary, zero_division=0)
        auc_roc = roc_auc_score(y_binary, y_pred_binary)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "auc_roc": auc_roc
        }

    def get_model(self):
        """
        Returns the internal LOF model.

        :return: Fitted LocalOutlierFactor model.
        """
        return self.model
