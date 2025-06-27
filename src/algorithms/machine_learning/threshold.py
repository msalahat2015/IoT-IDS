# threshold.py
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class ThresholdClassifier(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Threshold Classifier.

        :param params: Dictionary containing Threshold Classifier parameters.
                       Must include 'threshold'.
        """
        if params is None:
            params = {}
        self.params = params
        self.threshold = params.get('threshold')  # Get threshold, default is handled in train
        self.model = None  # No actual model is trained, but we keep it for consistency

    def train(self, X_train, y_train):
        """
        'Trains' the Threshold Classifier.  In reality, this checks for the
        existence of the threshold.

        :param X_train: Training data (not used for training).
        :param y_train: Training labels (not used for training).
        """
        if 'threshold' not in self.params:
            #If threshold is not provided, calculate the mean of the y_train.
            self.threshold = np.mean(y_train)
            print(f"Threshold not provided.  Setting threshold to: {self.threshold}")
        elif self.threshold is None:
            raise ValueError("ThresholdClassifier requires 'threshold' parameter in params.")
        self.model = True # Set model to true to indicate that the threshold is set

    def predict(self, X_test):
        """
        Makes predictions using the Threshold Classifier.

        :param X_test: Test data.  Assumes the input data is probabilities or scores.
        :return: Model predictions (0 or 1).
        """
        if self.model is None:
            raise Exception("The model must be trained first (threshold must be set).")
        return (X_test >= self.threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Threshold Classifier.

        :param X_test: Test data.
        :param y_test: True labels for the test data.
        :return: Dictionary containing evaluation results.
        """
        if self.model is None:
            raise Exception("The model must be trained first (threshold must be set).")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {"accuracy": accuracy, "classification_report": report}

    def get_model(self):
        """
        Returns the threshold value.
        """
        return self.threshold
