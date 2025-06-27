# weighted_svm.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import math
import pandas as pd
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class WeightedSVM(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Weighted Support Vector Machine classifier.
        This SVM allows you to specify custom class weights.

        :param params: Dictionary containing SVM parameters.  Must include
                       'class_weight' as a dictionary.
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None
        
    def get_weights(self, y_train):
        """
        Calculates class weights based on the distribution of classes in y_train.

        Args:
            y_train: The target variable (class labels).  Expected to be a NumPy array or a Pandas Series.

        Returns:
            dict: A dictionary where keys are class labels and values are the calculated weights.
        """
        # Ensure y_train is a Pandas Series
        if isinstance(y_train, pd.Series):
            y_series = y_train  # No conversion needed
        else:
            y_series = pd.Series(y_train)  # Convert NumPy array to Series

        # Get counts of each class
        y_train_count = y_series.value_counts()

        # Calculate log10 of the counts
        y_log = y_train_count.apply(lambda x: math.log10(x))

        # Find the maximum log10 value
        y_log_max = y_log.max()

        # Calculate weights to balance classes
        y_log = y_log.apply(lambda x: math.pow(10, y_log_max - x))

        # Convert the resulting Series to a dictionary
        weight_dict = y_log.to_dict()

        return weight_dict

    def train(self, X_train, y_train):
        """
        Trains the Weighted SVM model.

        :param X_train: Training data.
        :param y_train: Training labels.
        """
        # if 'class_weight' not in self.params:
        #     raise ValueError("WeightedSVM requires 'class_weight' parameter in params.")
        self.model = SVC(**self.params)
        self.model.class_weight_ = self.get_weights(y_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Weighted SVM model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Weighted SVM model.

        :param X_test: Test data.
        :param y_test: True labels for the test data.
        :return: Dictionary containing evaluation results.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {"accuracy": accuracy, "classification_report": report}

    def get_model(self):
        """
        Returns the trained Weighted SVM model.
        """
        return self.model
    
   