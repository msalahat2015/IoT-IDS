import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from ..base.base_algorithm import BaseAlgorithm  # type: ignore


class LSTMModel(BaseAlgorithm):
    """
    A Long Short-Term Memory (LSTM) classifier.
    """

    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        self.model = None  # Will be built after label shape is known

    def _build_model(self):
        """
        Builds the LSTM model using TensorFlow Keras.

        :return: A compiled Keras model.
        """
        input_shape = tuple(self.params.get('input_shape', (1, 29)))
        units = self.params.get('units', 64)
        dropout_rate = self.params.get('dropout_rate', 0.2)
        optimizer_name = self.params.get('optimizer', 'adam')
        learning_rate = self.params.get('learning_rate', 0.001)
        activation = self.params.get('activation', 'tanh')
        return_sequences = self.params.get('return_sequences', True)

        num_classes = self.params.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be set before building the model.")

        model = Sequential()
        model.add(LSTM(units, activation=activation, input_shape=input_shape, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

        if return_sequences:
            model.add(LSTM(units, activation=activation, return_sequences=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(num_classes, activation='softmax'))

        if optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name  # You may handle other optimizers if needed

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        """
        Trains the model.

        :param X: Training features (NumPy array). Must be 3D: (samples, time_steps, features)
        :param y: Training labels (integer labels, will be one-hot encoded automatically)
        """
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)  # shape becomes (samples, 1, features)

        if y.ndim == 1:
            num_classes = len(np.unique(y))
            y = to_categorical(y, num_classes)
            self.params['num_classes'] = num_classes  # update
        else:
            num_classes = y.shape[1]
            self.params['num_classes'] = num_classes

        # Now that we know number of classes, build model
        self.model = self._build_model()

        epochs = self.params.get('epochs', 10)
        batch_size = self.params.get('batch_size', 32)
        validation_split = self.params.get('validation_split', 0.0)

        if validation_split > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predicted class probabilities (NumPy array).
        """
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (integer or one-hot encoded).
        :return: Dictionary of evaluation metrics.
        """
        y_pred_probs = self.predict(X)

        if y.ndim == 1:
            y_true = y
        else:
            y_true = np.argmax(y, axis=1)

        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        auc_roc = roc_auc_score(to_categorical(y_true, y_pred_probs.shape[1]), y_pred_probs, multi_class='ovr')

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "auc_roc": auc_roc
        }

    def get_model(self):
        """
        Returns the trained model.

        :return: The trained Keras model.
        """
        return self.model
