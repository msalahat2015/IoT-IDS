# autoencoder.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, classification_report  # Import for potential use
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class

class Autoencoder(BaseAlgorithm):
    def __init__(self, params=None):
        """
        Initializes the Autoencoder.

        :param params: Dictionary containing Autoencoder parameters, including:
            - 'encoding_dim':  (required) The dimension of the encoded representation.
            - 'hidden_units': (optional) A list of hidden layer sizes.
            - 'epochs':       (optional) Number of training epochs.
            - 'batch_size':  (optional) Batch size for training.
            - 'activation': (optional) Activation function for the layers.
        """
        if params is None:
            params = {}
        self.params = params
        self.encoding_dim = params.get('encoding_dim')
        self.hidden_units = params.get('hidden_units', [128, 64])  # Default hidden layers
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 32)
        self.activation = params.get('activation', 'relu')
        self.model = None
        self.encoder = None  # Store the encoder model
        self.input_dim = None

        if self.encoding_dim is None:
            raise ValueError("Autoencoder requires 'encoding_dim' parameter.")

    def build_model(self, input_dim):
        """
        Builds the Autoencoder model.

        :param input_dim: The dimension of the input data.
        """
        self.input_dim = input_dim
        input_layer = Input(shape=(input_dim,))
        x = input_layer

        # Encoder
        for units in self.hidden_units:
            x = Dense(units, activation=self.activation)(x)
        encoded = Dense(self.encoding_dim, activation=self.activation)(x)
        self.encoder = Model(input_layer, encoded) #save encoder

        # Decoder
        x = encoded
        for units in reversed(self.hidden_units):
            x = Dense(units, activation=self.activation)(x)
        decoded = Dense(input_dim, activation='linear')(x)  # Use 'linear' for output

        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')  # Use MSE loss for autoencoders

    def train(self, X_train, y_train=None):
        """
        Trains the Autoencoder model.

        :param X_train: Training data.
        :param y_train:  Ignored, included for consistency with other models.
        """
        if self.model is None:
            self.build_model(X_train.shape[1])  # Build model with input dimension
        self.model.fit(X_train, X_train,  # Autoencoders reconstruct the input
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       verbose=0)

    def predict(self, X_test):
        """
        Makes predictions using the trained Balanced SVM model.

        :param X_test: Test data.
        :return: Model predictions.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X_test)
    def transform(self, X):
        """
        Encodes the input data using the trained encoder.

        :param X: Data to encode.
        :return: Encoded data.
        """
        if self.encoder is None:
            raise Exception("The model must be trained first.")
        return self.encoder.predict(X)

    def reconstruct(self, X):
        """
        Reconstructs the input data using the trained autoencoder.

        :param X: Data to reconstruct.
        :return: Reconstructed data.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test=None):
        """
        Evaluates the Autoencoder model by calculating the Mean Squared Error (MSE).

        :param X_test: Test data.
        :param y_test: Ignored, included for consistency.
        :return: Dictionary containing the MSE.
        """
        if self.model is None:
            raise Exception("The model must be trained first.")
        reconstructed = self.reconstruct(X_test)
        mse = tf.reduce_mean(tf.square(X_test - reconstructed)).numpy()
        return {"mse": mse}

    def get_model(self):
        """
        Returns the trained Autoencoder model.
        """
        return self.model
