# pca.py
from sklearn.decomposition import PCA
import numpy as np
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
class PCAWrapper(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the PCA wrapper.

        :param params: Dictionary containing PCA parameters.
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None
        self.n_components = params.get('n_components', None) # get n_components

    def train(self, X_train, y_train=None):
        """
        Trains the PCA model (performs dimensionality reduction).

        :param X_train: Training data.
        :param y_train:  Ignored, included for consistency with other models.
        """
        self.model = PCA(**self.params)
        self.model.fit(X_train)
        
    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predictions (NumPy array).
        """
        return self.model.predict(X)  # Return the predictions made by the model

    def transform(self, X):
        """
        Applies dimensionality reduction to the input data.

        :param X: Data to transform.
        :return: Transformed data.
        """
        if self.model is None:
            raise Exception("The PCA model must be trained first.")
        return self.model.transform(X)

    def get_model(self):
        """
        Returns the trained PCA model.
        """
        return self.model

    def get_n_components(self):
        """
        Returns the number of components.
        """
        return self.n_components
