# naive_bayes.py

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from ..base.base_algorithm import BaseAlgorithm  # type: ignore

class NaiveBayes(BaseAlgorithm):
    def __init__(self, params=None, nb_type='gaussian'):
        """
        Initializes the Naive Bayes classifier.

        :param params: Dictionary containing Naive Bayes parameters (can be empty).
        :param nb_type: Type of Naive Bayes ('gaussian' or 'multinomial').
        """
        if params is None:
            params = {}
        self.params = params
        self.nb_type = nb_type.lower()
        self.model = None
        self.scaler = None  # Used for MinMax scaling in MultinomialNB

    def train(self, X_train, y_train):
        """
        Trains the Naive Bayes model.

        :param X_train: Training data.
        :param y_train: Training labels.
        """
        if self.nb_type == 'multinomial':
            # Scale features to non-negative values for MultinomialNB
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X_train)
            self.model = MultinomialNB(**self.params)
        elif self.nb_type == 'gaussian':
            self.model = GaussianNB(**self.params)
        else:
            raise ValueError("Invalid Naive Bayes type. Must be 'gaussian' or 'multinomial'.")

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Naive Bayes model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")

        if self.nb_type == 'multinomial' and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Naive Bayes model.

        :param X_test: Test data.
        :param y_test: True labels for the test data.
        :return: Dictionary containing evaluation results.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def get_model(self):
        """
        Returns the trained Naive Bayes model.
        """
        return self.model
