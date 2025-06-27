import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore


class ANN(BaseAlgorithm):
    """Artificial Neural Network (ANN)."""

    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        self.model = self._build_model()
        self.one_hot_encoder = None

    def _build_model(self):
        """Builds the ANN model based on the provided parameters."""
        input_dim = self.params.get("input_dim")
        hidden_units = self.params.get("hidden_units", [64, 32])
        output_dim = self.params.get("output_dim")
        activation = self.params.get("activation", "relu")
        optimizer = self.params.get("optimizer", "adam")
        loss = self.params.get("loss", "categorical_crossentropy")

        if input_dim is None or output_dim is None:
            return None # Model building deferred until training

        model = Sequential()
        model.add(Dense(hidden_units[0], activation=activation, input_dim=input_dim))
        for units in hidden_units[1:]:
            model.add(Dense(units, activation=activation))
        model.add(Dense(output_dim, activation="softmax")) # Assuming multi-class

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=10, batch_size=32, validation_data=None, verbose=1):
        """
        Trains the ANN model.

        Args:
            X (numpy.ndarray): Training features.
            y (numpy.ndarray): Training labels.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            validation_data (tuple, optional): Tuple (X_val, y_val) for validation. Defaults to None.
            verbose (int, optional): Verbosity level during training. Defaults to 1.

        Returns:
            tensorflow.keras.callbacks.History: Training history object.
        """
        input_dim = X.shape[1]
        num_classes = self.params.get("num_classes")
        output_dim = self.params.get("output_dim")

        if num_classes is None and output_dim is None:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False)
            y_encoded = self.one_hot_encoder.fit_transform(y.reshape(-1, 1))
            self.params["num_classes"] = y_encoded.shape[1]
            self.params["output_dim"] = self.params["num_classes"]
        elif num_classes is not None:
            self.params["output_dim"] = num_classes
            self.one_hot_encoder = OneHotEncoder(sparse_output=False)
            y_encoded = self.one_hot_encoder.fit_transform(y.reshape(-1, 1))
        else:
            y_encoded = y # Assuming already encoded or binary

        if self.model is None or self.model.input_shape[-1] != input_dim or self.model.output_shape[-1] != self.params["output_dim"]:
            self.params["input_dim"] = input_dim
            self.model = self._build_model()
            if self.model is None:
                raise ValueError("Could not build the model. Ensure input_dim and output_dim are provided or can be inferred.")

        history = self.model.fit(X, y_encoded, epochs=epochs, batch_size=batch_size,
                                  validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, X):
        """
        Predicts class labels for the given input data.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        probabilities = self.model.predict(X)
        if self.one_hot_encoder:
            return np.argmax(probabilities, axis=1)
        else:
            return np.round(probabilities) # Assuming binary or regression if no encoder

    def evaluate(self, X, y):
        """
        Evaluates the trained model on the given data.

        Args:
            X (numpy.ndarray): Evaluation features.
            y (numpy.ndarray): Evaluation labels.

        Returns:
            dict: A dictionary containing the evaluation metrics (e.g., loss, accuracy).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.one_hot_encoder:
            y_encoded = self.one_hot_encoder.transform(y.reshape(-1, 1))
        else:
            y_encoded = y
        loss, accuracy = self.model.evaluate(X, y_encoded, verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    def get_model(self):
        """
        Returns the underlying Keras model.

        Returns:
            tensorflow.keras.models.Sequential: The Keras Sequential model.
        """
        return self.model

if __name__ == '__main__':
    # Example Usage with parameters
    X = np.random.rand(100, 29)
    y_multiclass = np.random.randint(0, 4, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

    ann_params = {
        "input_dim": 29,
        "hidden_units": [128, 64, 32],
        "output_dim": 4,
        "activation": "relu",
        "optimizer": "adam",
        "loss": "categorical_crossentropy"
    }

    ann_classifier = ANN(params=ann_params)
    history = ann_classifier.train(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
    y_pred = ann_classifier.predict(X_test)
    evaluation = ann_classifier.evaluate(X_test, y_test)
    print("\nEvaluation:", evaluation)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Example without explicitly providing input/output dim in params initially
    X_binary = np.random.rand(100, 10)
    y_binary = np.random.randint(0, 2, 100)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

    ann_binary_params = {
        "hidden_units": [32],
        "activation": "sigmoid",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "output_dim": 1 # For binary classification
    }

    ann_binary_classifier = ANN(params=ann_binary_params)
    history_binary = ann_binary_classifier.train(X_train_bin, y_train_bin, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
    y_pred_binary_prob = ann_binary_classifier.predict(X_test_bin)
    y_pred_binary = np.round(y_pred_binary_prob).flatten().astype(int)
    evaluation_binary = ann_binary_classifier.evaluate(X_test_bin, y_test_bin)
    print("\nBinary Evaluation:", evaluation_binary)
    print("\nBinary Classification Report:\n", classification_report(y_test_bin, y_pred_binary))
    print("\nBinary ROC AUC Score:", roc_auc_score(y_test_bin, y_pred_binary_prob))