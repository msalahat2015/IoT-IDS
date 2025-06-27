class BaseAlgorithm:
    """
    Base class for all algorithms.
    Defines the common interface that all algorithms should adhere to.
    """
    def __init__(self, params=None):
        """
        Initializes the algorithm.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None  # The actual model will be assigned in derived classes

    def train(self, X, y):
        """
        Trains the model.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array).
        """
        raise NotImplementedError("The train method must be implemented in the derived class.")

    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predictions (NumPy array).
        """
        raise NotImplementedError("The predict method must be implemented in the derived class.")

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
        :return: Evaluation metrics (dictionary).
        """
        raise NotImplementedError("The evaluate method must be implemented in the derived class.")

    def get_model(self):
        """
        Returns the trained model.

        :return: The trained model.
        """
        return self.model
