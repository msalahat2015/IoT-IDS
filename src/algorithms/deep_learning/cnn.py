import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # Import for one-hot encoding
import pandas as pd  # Import pandas

class CNNModel(BaseAlgorithm):
    """
    A 1D Convolutional Neural Network (CNN) classifier. This is designed for
    sequence data, where CNNs can identify local patterns.
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
        Builds the 1D CNN model using TensorFlow Keras (handling variable length).

        :return: A compiled Keras model.
        """
        # Default values
        input_shape = self.params.get('input_shape', (None, 1))  # Adjusted based on params
        filters = self.params.get('filters', 64)
        kernel_size = self.params.get('kernel_size', 3)
        pooling_size = self.params.get('pooling_size', 2)
        dropout_rate = self.params.get('dropout_rate', 0.2)
        optimizer_name = self.params.get('optimizer', 'adam')
        learning_rate = self.params.get('learning_rate', 0.001)
        num_classes = self.params.get('num_classes', 4)  # Based on the error
        num_conv_layers = self.params.get('num_conv_layers', 2) # number of conv layers
        padding_type = self.params.get('padding', 'valid').lower() # Get padding type from params

        model = Sequential()
        # Convolutional layers
        model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=input_shape, padding=padding_type))
        model.add(MaxPooling1D(pooling_size))
        model.add(Dropout(dropout_rate))

        for i in range(num_conv_layers - 1):
            filters *= 2
            model.add(Conv1D(filters, kernel_size, activation='relu', padding=padding_type))
            model.add(MaxPooling1D(pooling_size))
            model.add(Dropout(dropout_rate))

        model.add(GlobalMaxPooling1D())  # Use GlobalMaxPooling1D for variable length
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        # Use the learning rate
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name  # Or other optimizers

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        """
        Trains the model. Attempts to reshape X into 3D if it's 2D and the
        number of features is compatible with num_time_steps and features_per_step.
        y is now one-hot encoded.

        :param X: Training features (NumPy array or Pandas DataFrame).
        :param y: Training target variable (NumPy array of integer labels).
        """
        epochs = self.params.get('epochs', 1)  # Reduced for brevity
        batch_size = self.params.get('batch_size', 32)
        validation_split = self.params.get('validation_split', 0.2)
        num_time_steps = self.params.get('num_time_steps', 4)
        features_per_step = self.params.get('features_per_step', 10)
        num_classes = self.params.get('num_classes', 4) # Ensure num_classes is consistent

        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array

        if len(X.shape) == 2:
            total_features = X.shape[1]
            expected_features = num_time_steps * features_per_step
            if total_features >= expected_features and total_features % features_per_step == 0:
                print(f"Reshaping input data from {X.shape} to (-1, {num_time_steps}, {features_per_step})")
                X = X[:, :expected_features].reshape(-1, num_time_steps, features_per_step)
            else:
                print(f"Warning: Input data shape {X.shape} is not compatible with the expected sequence length ({num_time_steps} time steps * {features_per_step} features = {expected_features}). Ensure your input data has the correct sequential structure or adjust 'num_time_steps' and 'features_per_step' parameters.")
                print("Continuing with the input data in its current 2D shape, but assuming the last dimension represents features for the 1D CNN.")
                X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape to (batch_size, timesteps/features, 1 feature)
        elif len(X.shape) == 3:
            print(f"Input data already has 3 dimensions: {X.shape}")
            # Assuming the shape is (batch_size, time_steps, features)
            pass
        else:
            raise ValueError(f"Input data X should be 2D or 3D, but got shape {X.shape}")

        # One-hot encode the target variable
        y_encoded = to_categorical(y, num_classes=num_classes)

        if validation_split > 0.0:
            X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
                X, y_encoded, test_size=validation_split, random_state=42
            )
            self.model.fit(
                X_train,
                y_train_encoded,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val_encoded),
            )
        else:
            self.model.fit(X, y_encoded, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """
        Makes predictions. Returns class probabilities.

        :param X: Test features (NumPy array or Pandas DataFrame).
        :return: Predicted class probabilities (NumPy array).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array

        if len(X.shape) == 2:
            num_time_steps = self.params.get('num_time_steps', 4)
            features_per_step = self.params.get('features_per_step', 10)
            total_features = X.shape[1]
            expected_features = num_time_steps * features_per_step
            if total_features >= expected_features and total_features % features_per_step == 0:
                print(f"Reshaping prediction input from {X.shape} to (-1, {num_time_steps}, {features_per_step}) for prediction.")
                X = X[:, :expected_features].reshape(-1, num_time_steps, features_per_step)
            else:
                print("Warning: Prediction input data is not in the expected 3D sequential format. Assuming the last dimension represents features for the 1D CNN.")
                X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape to (batch_size, timesteps/features, 1 feature)
        elif len(X.shape) != 3:
            raise ValueError(f"Prediction input X should be 2D or 3D, but got shape {X.shape}")
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array or Pandas DataFrame).
        :param y: Test target variable (NumPy array of integer labels).
        :return: Evaluation metrics (dictionary).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array

        y_pred_probs = self.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(to_categorical(y, num_classes=self.params.get('num_classes', 4)), axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        auc_roc = roc_auc_score(
            y_true, y_pred_probs, multi_class='ovr'
        )  # Use 'ovr' for multi-class

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}

    def get_model(self):
        """
        Returns the trained model.

        :return: The trained model
        """
        return self.model