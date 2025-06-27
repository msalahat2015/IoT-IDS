import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore
from sklearn.model_selection import train_test_split # Added for clarity


class DNNModel(BaseAlgorithm):
    """
    A Deep Neural Network (DNN) classifier.
    """

    def __init__(self, params=None):
        """
        Initializes the classifier.

        :param params: Algorithm parameters (dictionary).
        """
        if params is None:
            params = {}
        self.params = params
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the DNN model using TensorFlow Keras.

        :return: A compiled Keras model.
        """
        # Default values
        input_shape = self.params.get('input_shape', (10,))  # (num_features,)
        hidden_units = self.params.get('hidden_units', [128, 64])  # List of hidden unit sizes
        dropout_rate = self.params.get('dropout_rate', 0.2)
        optimizer_name = self.params.get('optimizer', 'adam')
        learning_rate = self.params.get('learning_rate', 0.001)
        num_classes = self.params.get('num_classes', 2)

        model = Sequential()
        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))  # Use Input layer explicitly
        model.add(Dense(hidden_units[0], activation='relu'))
        model.add(Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))

        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification
            loss_function = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class
            loss_function = 'categorical_crossentropy'

        # Use the learning rate
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name  # Or handle other optimizers

        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        return model
    
    def _build_model1(self):
        """
        Builds the DNN model using TensorFlow Keras.

        :return: A compiled Keras model.
        """
        # Default values
        input_shape = self.params.get('input_shape', (10,))  # (num_features,)
        hidden_units = self.params.get('hidden_units', [128, 64])  # List of hidden unit sizes
        dropout_rate = self.params.get('dropout_rate', 0.2)
        optimizer = self.params.get('optimizer', 'adam')
        learning_rate = self.params.get('learning_rate', 0.001)
        num_classes = self.params.get('num_classes', 2)

        model = Sequential()
        # Input layer
        model.add(Dense(hidden_units[0], activation='relu', input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class

        # Use the learning rate
        if optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer  # Or other optimizers

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # Use categorical_crossentropy
        return model

    def train(self, X, y):
        """
        Trains the model.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array). y should be one-hot encoded.
        """
        epochs = self.params.get('epochs', 10)
        batch_size = self.params.get('batch_size', 32)
        validation_split = self.params.get('validation_split', 0.0)
        
        num_classes = self.params.get('num_classes', 2) # Get num_classes
        # One-hot encode y if it's not binary classification
        if num_classes > 2:
            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y.reshape(-1, 1))

        if validation_split > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """
        Makes predictions.  Returns class probabilities.

        :param X: Test features (NumPy array).
        :return: Predicted class probabilities (NumPy array).
        """
        return self.model.predict(X)

    def evaluate1(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array). y should be one-hot encoded.
        :return: Evaluation metrics (dictionary).
        """
        y_pred_probs = self.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}
    def evaluate(self, X, y):
        """
        Evaluates the model's performance.  Handles both binary and multi-class.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
                    For binary, y should be 1D. For multi-class, y should be one-hot encoded.
        :return: Evaluation metrics (dictionary).
        """
        
        y_pred_probs = self.predict(X)  # Get probabilities
        num_classes = self.params.get('num_classes', 2) # Get num_classes
        
        # One-hot encode y if it's not binary classification
        if num_classes > 2:
            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y.reshape(-1, 1))

        if num_classes == 2:
            # Binary classification
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Threshold and flatten
            y_true = y.flatten()  # Ensure y is 1D
            try:
                auc_roc = roc_auc_score(y_true, y_pred_probs.flatten())
            except ValueError as e:
                print(f"Warning: Could not calculate AUC-ROC for binary case: {e}")
                auc_roc = np.nan
        else:
            # Multi-class classification
            y_pred = np.argmax(y_pred_probs, axis=1)  # Get class predictions
            y_true = np.argmax(y, axis=1)  # Get true class labels from one-hot
            auc_roc = roc_auc_score(y, y_pred_probs, multi_class='ovr',
                                    average='macro')  # AUC-ROC for multi-class

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)  # Handle zero division

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}

    def evaluate1(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
        :return: Evaluation metrics (dictionary).
        """
        y_pred_probs = self.predict(X)
        y_pred = (y_pred_probs > 0.5).astype(int)  # For binary, threshold at 0.5

        # Ensure y_true is 1D for binary classification
        y_true = y.flatten()

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        # Calculate AUC-ROC for binary case
        if self.params.get('num_classes', 2) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_pred_probs)
            except ValueError as e:
                print(f"Warning: Could not calculate AUC-ROC for binary class: {e}")
                auc_roc = np.nan
        else:
            auc_roc = np.nan

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}
    
    def get_model(self):
        """
        Returns the trained model.

        :return: The trained model
        """
        return self.model
