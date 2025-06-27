# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from ..base.base_algorithm import BaseAlgorithm  # type: ignore
# import inspect
# import json
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, LSTM, Bidirectional
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical


# class StackingClassifier(BaseAlgorithm):
#     """
#     A Stacking Classifier that can handle multiple layers of stacking, including deep BiLSTM models.
#     """

#     def __init__(self, params=None):
#         """
#         Initializes the Stacking Classifier.

#         :param params: Algorithm parameters (dictionary or path to a JSON file).
#         """
#         if isinstance(params, str):  # If params is a string, assume it's a file path
#             try:
#                 with open(params, 'r') as f:
#                     self.params = json.load(f)
#             except FileNotFoundError:
#                 raise FileNotFoundError(f"File not found at {params}.")
#             except json.JSONDecodeError:
#                 raise json.JSONDecodeError(f"Invalid JSON format in {params}.")
#         elif params is None:
#             self.params = {}
#         else:
#             self.params = params

#         self.layers = self.params.get('layers', [])
#         self.meta_model_params = self.params.get('meta_model_params', {})
#         self.cv_folds = self.params.get('cv_folds', 5)
#         self.one_hot_encode = self.params.get('one_hot_encode', True)
#         self.estimators = self._initialize_estimators()
#         self.meta_model = self._build_meta_model()
#         self.trained_models_per_layer = []  # To store trained models

#     def _import_model(self, model_name):
#         """
#         Dynamically imports a model class.

#         :param model_name: The class path of the model (e.g., 'sklearn.linear_model.LogisticRegression').
#         :return: The model class.
#         :raises ValueError: If the model cannot be imported.
#         """
#         try:
#             module_name, class_name = model_name.rsplit('.', 1)
#             module = __import__(module_name, fromlist=[class_name])
#             model_class = getattr(module, class_name)
#             return model_class
#         except ImportError as e:
#             raise ValueError(
#                 f"Could not import model {model_name}. Check the module path and dependencies.  Error: {e}"
#             )
#         except AttributeError as e:
#             raise ValueError(
#                 f"Could not find class {class_name} in module {module_name}. Error: {e}"
#             )
#         except Exception as e:
#             raise ValueError(f"Error importing model {model_name}: {e}")

#     def _initialize_estimators(self):
#         """
#         Initializes the estimators for each layer based on the parameters in self.params.
#         Handles different model types, including BiLSTM.

#         :return: A list of lists of (name, model) tuples, where each inner list represents a layer.
#         """
#         if not self.layers:
#             raise ValueError("No layers provided in parameters.")

#         all_estimators = []
#         for layer_params in self.layers:
#             layer_estimators = []
#             for est_param in layer_params.get('estimators', []):
#                 name = est_param.get('name')
#                 model_name = est_param.get('model')
#                 model_params = {
#                     k: v
#                     for k, v in est_param.items()
#                     if k not in ['name', 'model', 'layers', 'optimizer', 'loss', 'metrics', 'epochs', 'batch_size']
#                 }  # Exclude special params

#                 if not name:
#                     raise ValueError("Each estimator must have a 'name'.")
#                 if not model_name:
#                     raise ValueError("Each estimator must have a 'model' class path.")

#                 # Handle TensorFlow Keras models (including BiLSTM)
#                 if model_name.startswith('tensorflow'):
#                     model = self._build_tf_model(est_param)  # Build Keras model
#                 else:
#                     # Handle scikit-learn models
#                     model_class = self._import_model(model_name)
#                     try:
#                         valid_params = list(inspect.signature(model_class).parameters.keys())
#                         invalid_params = [
#                             k for k in model_params if k not in valid_params
#                         ]
#                         if invalid_params:
#                             raise ValueError(
#                                 f"Invalid parameters {invalid_params} for model {model_name}."
#                             )
#                         model = model_class(**model_params)
#                     except Exception as e:
#                         raise ValueError(f"Could not initialize model {model_name}: {e}")

#                 layer_estimators.append((name, model))
#             all_estimators.append(layer_estimators)
#         return all_estimators

#     def _build_tf_model(self, est_param):
#         """
#         Builds a TensorFlow Keras model based on the provided parameters,
#         including handling Bidirectional LSTM.

#         :param est_param:  The estimator parameter dictionary.
#         :return: A compiled TensorFlow Keras model.
#         """
#         model_name = est_param.get('model')
#         layers_config = est_param.get('layers', [])
#         optimizer_config = est_param.get('optimizer', {})
#         loss = est_param.get('loss', 'categorical_crossentropy')
#         metrics = est_param.get('metrics', ['accuracy'])

#         model = Sequential()
#         for layer_config in layers_config:
#             class_name = layer_config.get('class_name')
#             config = layer_config.get('config', {})

#             if class_name == 'tensorflow.keras.layers.Bidirectional':
#                 # Handle BiLSTM and Deep BiLSTM
#                 lstm_config = config.get('layer', {})
#                 if isinstance(lstm_config, list):  # Check for a list of LSTM layers (for deep BiLSTM)
#                     for lstm_layer_config in lstm_config:
#                         lstm_units = lstm_layer_config.get('config', {}).get('units', 64)
#                         lstm_activation = lstm_layer_config.get('config', {}).get('activation', 'relu')
#                         lstm_return_sequences = lstm_layer_config.get('config', {}).get('return_sequences', False)
#                         lstm_input_shape = lstm_layer_config.get('config', {}).get('input_shape')

#                         if lstm_input_shape:
#                             model.add(
#                                 Bidirectional(
#                                     LSTM(
#                                         lstm_units,
#                                         activation=lstm_activation,
#                                         return_sequences=lstm_return_sequences,
#                                         input_shape=lstm_input_shape,
#                                     ),
#                                     merge_mode=config.get('merge_mode', 'concat'),
#                                 )
#                             )
#                         else:
#                             model.add(
#                                 Bidirectional(
#                                     LSTM(
#                                         lstm_units,
#                                         activation=lstm_activation,
#                                         return_sequences=lstm_return_sequences,
#                                     ),
#                                     merge_mode=config.get('merge_mode', 'concat'),
#                                 )
#                             )
#                 else:  # Handle single BiLSTM layer
#                     lstm_units = lstm_config.get('config', {}).get('units', 64)
#                     lstm_activation = lstm_config.get('config', {}).get('activation', 'relu')
#                     lstm_return_sequences = lstm_config.get('config', {}).get('return_sequences', False)
#                     lstm_input_shape = lstm_config.get('config', {}).get('input_shape')
#                     if lstm_input_shape:
#                         model.add(
#                             Bidirectional(
#                                 LSTM(
#                                     lstm_units,
#                                     activation=lstm_activation,
#                                     return_sequences=lstm_return_sequences,
#                                     input_shape=lstm_input_shape,
#                                 ),
#                                 merge_mode=config.get('merge_mode', 'concat'),
#                             )
#                         )
#                     else:
#                         model.add(
#                             Bidirectional(
#                                 LSTM(
#                                     lstm_units,
#                                     activation=lstm_activation,
#                                     return_sequences=lstm_return_sequences,
#                                 ),
#                                 merge_mode=config.get('merge_mode', 'concat'),
#                             )
#                         )
#             elif class_name == 'tensorflow.keras.layers.LSTM':
#                 # Handle standard LSTM
#                 units = config.get('units', 64)
#                 activation = config.get('activation', 'relu')
#                 return_sequences = config.get('return_sequences', False)
#                 input_shape = config.get('input_shape')
#                 if input_shape:
#                     model.add(
#                         LSTM(
#                             units,
#                             activation=activation,
#                             return_sequences=return_sequences,
#                             input_shape=input_shape,
#                         )
#                     )
#                 else:
#                     model.add(LSTM(units, activation=activation, return_sequences=return_sequences))
#             elif class_name == 'tensorflow.keras.layers.Dense':
#                 # Handle Dense layers
#                 units = config.get('units', 128)
#                 activation = config.get('activation', 'relu')
#                 input_shape = config.get('input_shape')
#                 if input_shape:
#                     model.add(Dense(units, activation=activation, input_shape=input_shape))
#                 else:
#                     model.add(Dense(units, activation=activation))
#             else:
#                 raise ValueError(f"Unsupported layer type: {class_name}")

#         # Optimizer
#         optimizer_name = optimizer_config.get('class_name', 'Adam')
#         learning_rate = optimizer_config.get('config', {}).get('learning_rate', 0.001)
#         if optimizer_name == 'Adam':
#             optimizer = Adam(learning_rate=learning_rate)
#         else:
#             raise ValueError(f"Unsupported optimizer: {optimizer_name}")

#         model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#         return model

#     def _build_meta_model(self):
#         """
#         Builds the meta-model (the final estimator) using TensorFlow Keras.
#         This meta-model takes the predictions from the previous layer(s) as input.
#         """
#         # Default meta-model parameters
#         meta_input_dim = self.layers[-1].get('estimators')  # Number of estimators in last layer
#         if not meta_input_dim:
#             raise ValueError("The last layer should have at least one estimator")
#         meta_hidden_units = self.meta_model_params.get('hidden_units', [128, 64, 32])  # More layers
#         meta_dropout_rate = self.meta_model_params.get('dropout_rate', 0.2)
#         meta_optimizer = self.meta_model_params.get('optimizer', 'adam')
#         meta_learning_rate = self.meta_model_params.get('learning_rate', 0.001)
#         num_classes = self.params.get('num_classes', 2)  # get num classes

#         meta_input = Input(shape=(meta_input_dim,), name='meta_input')
#         x = meta_input

#         for units in meta_hidden_units:
#             x = Dense(units, activation='relu')(x)
#             x = Dropout(meta_dropout_rate)(x)
#         meta_output = Dense(num_classes, activation='softmax')(x)  # Use softmax for multi-class

#         meta_model = Model(inputs=meta_input, outputs=meta_output)

#         if meta_optimizer.lower() == 'adam':
#             optimizer = Adam(learning_rate=meta_learning_rate)
#         else:
#             optimizer = meta_optimizer

#         meta_model.compile(
#             optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']
#         )
#         return meta_model

#     def _train_layer(self, X, y, layer_estimators, cv_folds):
#         """
#         Trains the estimators for a single layer using cross-validation.

#         :param X: Training features (NumPy array).
#         :param y: Training target variable (NumPy array).
#         :param layer_estimators: List of (name, model) tuples for the layer.
#         :param cv_folds: Number of cross-validation folds.
#         :return:  Tuple containing:
#             - predictions from the layer (NumPy array, shape: (n_samples, n_estimators))
#             - trained layer models
#         """
#         n_samples = X.shape[0]
#         n_estimators = len(layer_estimators)
#         # Initialize an empty array to hold the predictions from all the models in the layer.
#         layer_predictions = np.zeros((n_samples, n_estimators))
#         # Will store the trained models.
#         trained_layer_models = []

#         kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
#         for i, (train_index, val_index) in enumerate(kf.split(X, y)):
#             X_train, X_val = X[train_index], X[val_index]
#             y_train, y_val = y[train_index], y[val_index]

#             for j, (name, model) in enumerate(layer_estimators):
#                 model.fit(X_train, y_train)
#                 # Store the trained model
#                 trained_layer_models.append((name, model))
#                 # Predict on the validation set.
#                 y_pred = model.predict(X_val)
#                 if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
#                     # For multi-class, get probabilities for each class
#                     layer_predictions[val_index, j] = y_pred
#                 else:
#                     # For binary or single-class, ensure y_pred is 1D
#                     layer_predictions[val_index, j] = np.ravel(y_pred)

#         return layer_predictions, trained_layer_models

#     def train(self, X, y):
#         """
#         Trains the Stacking Classifier.

#         :param X: Training features (NumPy array).
#         :param y: Training target variable (NumPy array).
#         """
#         if self.one_hot_encode:
#             y = to_categorical(y)

#         n_classes = y.shape[1] if len(y.shape) > 1 else 1
#         self.params['num_classes'] = n_classes

#         # Initialize a list to store the predictions from each layer
#         layer_predictions_list = []
#         trained_models_per_layer = []

#         # Train each layer
#         for i, layer_estimators in enumerate(self.estimators):
#             print(f"Training Layer {i + 1}/{len(self.estimators)}...")
#             layer_predictions, trained_layer_models = self._train_layer(
#                 X, y, layer_estimators, self.cv_folds
#             )
#             layer_predictions_list.append(layer_predictions)
#             trained_models_per_layer.append(trained_layer_models)
#             # Use the predictions from the current layer as features for the next layer
#             X = layer_predictions
#         self.trained_models_per_layer = trained_models_per_layer  # store

#         # Train the meta-model on the predictions from the last layer
#         print("Training Meta-Model...")
#         self.meta_model.fit(
#             X,
#             y,
#             epochs=self.meta_model_params.get('epochs', 10),
#             batch_size=self.meta_model_params.get('batch_size', 32),
#             validation_split=self.meta_model_params.get('validation_split', 0.0),
#         )

#     def predict(self, X):
#         """
#         Makes predictions using the trained Stacking Classifier.

#         :param X: Test features (NumPy array).
#         :return: Predictions (NumPy array).
#         """
#         # if self.one_hot_encode: # removed one hot encode from predict
#         #    y = to_categorical(y)

#         # Get predictions from each layer
#         layer_predictions_list = []
#         for i, layer_estimators in enumerate(self.trained_models_per_layer):
#             layer_predictions = np.zeros((X.shape[0], len(layer_estimators)))
#             for j, (name, model) in enumerate(layer_estimators):
#                 y_pred = model.predict(X)
#                 if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
#                     layer_predictions[:, j] = y_pred
#                 else:
#                     layer_predictions[:, j] = np.ravel(y_pred)
#             layer_predictions_list.append(layer_predictions)
#             X = layer_predictions
#         # Get predictions from the meta-model
#         meta_predictions = self.meta_model.predict(X)
#         return meta_predictions

#     def evaluate(self, X, y):
#         """
#         Evaluates the model's performance.

#         :param X: Test features (NumPy array).
#         :param y: Test target variable (NumPy array).
#         :return: Evaluation metrics (dictionary).
#         """
#         if self.one_hot_encode:
#             y = to_categorical(y)
#         y_pred_probs = self.predict(X)
#         y_pred = np.argmax(y_pred_probs, axis=1)
#         y_true = np.argmax(y, axis=1)

#         accuracy = accuracy_score(y_true, y_pred)
#         report = classification_report(y_true, y_pred)
#         auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')

#         return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}

#     def get_model(self):
#         """
#         Returns the trained model.

#         :return: The trained model
#         """
#         return self.meta_model  # changed

#     def get_estimators(self):
#         """
#         Returns the list of estimators (models) used in the Stacking Classifier.

#         :return: A list of lists of (name, model) tuples.
#         """
#         return self.estimators


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional

def create_bilstm_model(input_shape, units=128, activation='tanh', recurrent_activation='sigmoid', output_units=10, output_activation='softmax'):
    """
    Creates a single-layer Bidirectional LSTM model.

    Args:
        input_shape: Shape of the input data (time steps, features).
        units:  Number of LSTM units in each direction.
        activation: Activation function for the cell state and hidden state.
        recurrent_activation: Activation function for the recurrent step.
        output_units: Number of units in the output Dense layer.
        output_activation: Activation function for the output Dense layer.

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 input_shape=input_shape),
                            # Defaults to 'concat', which is usually what you want.
                            merge_mode='concat'))
    model.add(Dense(output_units, activation=output_activation))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def create_deep_bilstm_model(input_shape, units=128, num_layers=2, activation='tanh', recurrent_activation='sigmoid',  output_units=10, output_activation='softmax'):
    """
    Creates a deep Bidirectional LSTM model with multiple stacked BiLSTM layers.

    Args:
        input_shape: Shape of the input data (time steps, features).
        units: Number of LSTM units in each direction for all layers.
        num_layers: The number of stacked BiLSTM layers.
        activation: Activation function for the cell state and hidden state.
        recurrent_activation: Activation function for the recurrent step.
        output_units: Number of units in the output Dense layer.
        output_activation: Activation function for the output Dense layer.

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    # Add the first BiLSTM layer, which needs the input_shape
    model.add(Bidirectional(LSTM(units,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 return_sequences=True, #important
                                 input_shape=input_shape),
                            merge_mode='concat'))

    # Add the intermediate BiLSTM layers
    for _ in range(num_layers - 2):
        model.add(Bidirectional(LSTM(units,
                                     activation=activation,
                                     recurrent_activation=recurrent_activation,
                                     return_sequences=True),  #important
                                merge_mode='concat'))
    # Add the final BiLSTM layer, which does not return sequences.
    model.add(Bidirectional(LSTM(units,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation),
                            merge_mode='concat'))
    model.add(Dense(output_units, activation=output_activation))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



if __name__ == '__main__':
    # Example usage:
    # Generate some dummy data
    time_steps = 20
    features = 10
    num_samples = 100
    num_classes = 5  # For example, if you have 5 classes

    X = tf.random.normal(shape=(num_samples, time_steps, features))
    # Create dummy one-hot encoded labels
    y = tf.keras.utils.to_categorical(tf.random.uniform(shape=(num_samples,), minval=0, maxval=num_classes, dtype=tf.int32), num_classes=num_classes)


    # 1. Create and train a single-layer BiLSTM model
    bilstm_model = create_bilstm_model(input_shape=(time_steps, features), units=64, output_units=num_classes)
    bilstm_model.summary()
    bilstm_history = bilstm_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # 2. Create and train a deep BiLSTM model
    deep_bilstm_model = create_deep_bilstm_model(input_shape=(time_steps, features), units=64, num_layers=3, output_units=num_classes)
    deep_bilstm_model.summary()
    deep_bilstm_history = deep_bilstm_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # You can then use the trained models for prediction:
    # predictions = bilstm_model.predict(X_test)
    # predictions_deep = deep_bilstm_model.predict(X_test)
