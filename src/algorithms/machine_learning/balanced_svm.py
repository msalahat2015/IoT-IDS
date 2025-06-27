# balanced_svm.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class BalancedSVM(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Balanced Support Vector Machine classifier.
        This SVM automatically adjusts weights to handle imbalanced classes.

        :param params: Dictionary containing SVM parameters.  It's recommended *not* to
                       include 'class_weight' here, as it's forced to 'balanced'.
        """
        if params is None:
            params = {}
        # Force class_weight to 'balanced' to handle class imbalance.
        self.params = {**params, 'class_weight': 'balanced'}
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the Balanced SVM model.

        :param X_train: Training data.
        :param y_train: Training labels.
        """
        self.model = SVC(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Balanced SVM model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the Balanced SVM model.

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
        Returns the trained Balanced SVM model.
        """
        return self.model
