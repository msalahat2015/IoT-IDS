# gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class GradientBoosting(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Gradient Boosting classifier.

        :param params: Dictionary containing Gradient Boosting parameters.
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the Gradient Boosting model.

        :param X_train: Training data.
        :param y_train: Training labels.
        """
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Gradient Boosting model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Gradient Boosting model.

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
        Returns the trained Gradient Boosting model.
        """
        return self.model
