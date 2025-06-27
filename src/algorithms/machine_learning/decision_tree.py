# decision_tree.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class DecisionTree(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Decision Tree classifier.

        :param params: Dictionary containing Decision Tree parameters.
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the Decision Tree model.

        :param X_train: Training data.
        :param y_train: Training labels.
        """
        self.model = DecisionTreeClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Decision Tree model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Decision Tree model.

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
        Returns the trained Decision Tree model.
        """
        return self.model
