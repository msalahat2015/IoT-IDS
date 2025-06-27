import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ..base.base_algorithm import BaseAlgorithm  # type: ignore
import inspect
import json
import joblib # Import joblib for saving scikit-learn models
import os # Import os for path manipulation

# Import TensorFlow components for saving/loading Keras models
from tensorflow.keras.models import Model, load_model # Added load_model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import warnings

# Import specific scikit-learn model classes for isinstance checks
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


class StackingClassifier(BaseAlgorithm):
    """
    A Stacking Classifier that can handle multiple layers of stacking.
    """

    def __init__(self, params=None):
        """
        Initializes the Stacking Classifier.

        :param params: Algorithm parameters (dictionary or path to a JSON file).
        """
        if isinstance(params, str):  # If params is a string, assume it's a file path
            try:
                with open(params, 'r') as f:
                    self.params = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found at {params}.")
            except json.JSONDecodeError:
                raise json.JSONDecodeError(f"Invalid JSON format in {params}.")
        elif params is None:
            self.params = {}
        else:
            self.params = params

        self.layers = self.params.get('layers', [])
        self.meta_model_params = self.params.get('meta_model_params', {})
        self.cv_folds = self.params.get('cv_folds', 5)  # Default number of cross-validation folds
        self.one_hot_encode = self.params.get('one_hot_encode', True) #whether to one hot encode y
        
        self.estimators = self._initialize_estimators()
        # self.meta_model will be built/re-built in the train method after num_classes is known
        self.meta_model = None 
        self.trained_models_per_layer = [] # To store models trained on full data
        self.num_classes = None # Store num_classes directly as an attribute

    def _import_model(self, model_name):
        """
        Dynamically imports a model class.

        :param model_name:  The class path of the model (e.g., 'sklearn.linear_model.LogisticRegression').
        :return: The model class.
        :raises ValueError: If the model cannot be imported.
        """
        try:
            module_name, class_name = model_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class
        except ImportError as e:
            raise ValueError(f"Could not import model {model_name}. Check the module path and dependencies.  Error: {e}")
        except AttributeError as e:
            raise ValueError(f"Could not find class {class_name} in module {module_name}. Error: {e}")
        except Exception as e:
            raise ValueError(f"Error importing model {model_name}: {e}")

    def _initialize_estimators(self):
        """
        Initializes the estimators for each layer based on the parameters in self.params.

        :return: A list of lists of (name, model) tuples, where each inner list represents a layer.
        """
        if not self.layers:
            raise ValueError("No layers provided in parameters.")

        all_estimators = []
        for layer_params in self.layers:
            layer_estimators = []
            for est_param in layer_params.get('estimators', []):
                name = est_param.get('name')
                model_name = est_param.get('model')
                model_params = {k: v for k, v in est_param.items() if k not in ['name', 'model']}

                if not name:
                    raise ValueError("Each estimator must have a 'name'.")
                if not model_name:
                    raise ValueError("Each estimator must have a 'model' class path.")

                model_class = self._import_model(model_name)

                try:
                    valid_params = list(inspect.signature(model_class).parameters.keys())
                    invalid_params = [k for k in model_params if k not in valid_params]
                    if invalid_params:
                        raise ValueError(
                            f"Invalid parameters {invalid_params} for model {model_name}."
                        )
                    model = model_class(**model_params)
                except Exception as e:
                    raise ValueError(f"Could not initialize model {model_name}: {e}")
                layer_estimators.append((name, model))
            all_estimators.append(layer_estimators)
        return all_estimators
    
    def _build_meta_model(self):
        """
        Generates the meta-model (the final estimator) using TensorFlow Keras.
        This meta-model takes the predictions from the previous layer(s) as input.
        """
        num_classes = self.num_classes
        if num_classes is None:
            raise ValueError("num_classes must be set before building the meta-model. Call train() first.")

        meta_input_dim = 0
        if not self.estimators:
            raise ValueError("No estimators initialized to determine meta-model input dimension.")

        # Calculate meta_input_dim based on the output of the last layer's estimators
        # This uses the original estimators definition to calculate the input dimension
        # as the trained models might not be available yet when this is called by train().
        for name, model_instance in self.estimators[-1]: # Iterate through estimators in the last layer's definition
            # Heuristic to determine output dimension for classification models:
            # If the model has 'predict_proba' and can output probabilities for multi-class,
            # its output dimension for stacking is num_classes.
            # Otherwise, it's typically 1 (for class labels or binary probabilities).
            
            # Check for predict_proba availability and type of model
            if hasattr(model_instance, 'predict_proba'):
                if isinstance(model_instance, (
                    LogisticRegression,
                    DecisionTreeClassifier,
                    RandomForestClassifier,
                    GradientBoostingClassifier
                )) or (isinstance(model_instance, SVC) and getattr(model_instance, 'probability', False)):
                    meta_input_dim += num_classes
                else:
                    meta_input_dim += 1 # Assume it contributes 1 feature
            else:
                meta_input_dim += 1
        
        if meta_input_dim == 0:
            raise ValueError("Could not determine meta_model input dimension. Ensure estimators are properly configured and num_classes is set.")

        meta_hidden_units = self.meta_model_params.get('hidden_units', [128, 64, 32])
        meta_dropout_rate = self.meta_model_params.get('dropout_rate', 0.2)
        meta_optimizer_name = self.meta_model_params.get('optimizer', 'adam')
        meta_learning_rate = self.meta_model_params.get('learning_rate', 0.001)

        meta_input = Input(shape=(meta_input_dim,), name='meta_input')
        x = meta_input

        for units in meta_hidden_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(meta_dropout_rate)(x)
        meta_output = Dense(num_classes, activation='softmax')(x) # Use softmax for multi-class

        meta_model = Model(inputs=meta_input, outputs=meta_output)

        if meta_optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=meta_learning_rate)
        else:
            optimizer = meta_optimizer_name # Keras will try to resolve this string

        meta_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return meta_model
    
    def _train_layer(self, X: pd.DataFrame, y: np.ndarray, layer_estimators: list, cv_folds: int):
        """
        Trains the estimators for a single layer using cross-validation.

        :param X: Training features (NumPy array or Pandas DataFrame).
        :param y: Training target variable (NumPy array - assumed one-hot encoded if needed).
        :param layer_estimators: List of (name, model) tuples for the layer.
        :param cv_folds: Number of cross-validation folds.
        :return:  Tuple containing:
            - predictions from the layer (NumPy array, shape: (n_samples, total_output_features_from_layer))
            - trained layer models (fitted on the full X, y)
        """
        n_samples = X.shape[0]
        num_classes = self.num_classes # Get num_classes from stored parameters

        # This list will store the predictions (probabilities or single values) from each estimator
        # across all folds, arranged to be stacked horizontally.
        oof_predictions_per_estimator = [np.zeros((n_samples, 0)) for _ in range(len(layer_estimators))]
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Convert y to labels for scikit-learn models if it's one-hot encoded
        y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

        trained_layer_models = [] # To store models trained on the full data for this layer

        for j, (name, model_template) in enumerate(layer_estimators):
            # Clone the model for each estimator to ensure independent training across folds/layers
            try:
                # For scikit-learn models, inspect their get_params method
                model = model_template.__class__(**model_template.get_params())
            except AttributeError:
                # Fallback for models without get_params or simple instantiation
                model = model_template.__class__() # This might lose initial params if not carefully handled
                warnings.warn(f"Model {name} does not have get_params(). Re-initializing without original parameters for CV folds. This might not be desired.")

            estimator_fold_preds = [] # To accumulate predictions for this specific estimator across folds
            
            for train_index, val_index in kf.split(X, y_labels): # Use y_labels for KFold split
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train_fold_labels = y_labels[train_index] # Use labels for fitting

                model.fit(X_train, y_train_fold_labels)
                
                y_pred_val = None
                # Determine if predict_proba should be used
                if hasattr(model, 'predict_proba') and \
                   (isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)) or \
                    (isinstance(model, SVC) and getattr(model, 'probability', False))):
                    y_pred_val = model.predict_proba(X_val)
                elif hasattr(model, 'predict'):
                    y_pred_val = model.predict(X_val)
                else:
                    raise RuntimeError(f"Estimator {name} does not have 'predict' or 'predict_proba' method.")
                
                # Ensure predictions are 2D
                if y_pred_val.ndim == 1:
                    y_pred_val = y_pred_val.reshape(-1, 1)
                
                # Store predictions in the correct place in the full OOF array
                if oof_predictions_per_estimator[j].shape[1] == 0: # First time setting the column count
                    oof_predictions_per_estimator[j] = np.zeros((n_samples, y_pred_val.shape[1]))
                oof_predictions_per_estimator[j][val_index] = y_pred_val
            
            # After all folds, fit this estimator on the full data for future predictions
            # The model variable here is the one used within the loop, which has been fitted on various folds.
            # We want to use the *cloned* model for the full training.
            # So, re-instantiate or ensure you're working with a fresh clone for final fit.
            # A cleaner way is to keep the original model_template and fit it on full data here.
            
            # Fit the original model_template on the full training data
            full_trained_model = model_template.__class__(**model_template.get_params())
            full_trained_model.fit(X, y_labels)
            trained_layer_models.append((name, full_trained_model))

        # Concatenate predictions from all estimators in this layer horizontally
        # Each element in oof_predictions_per_estimator is (n_samples, num_features_for_this_estimator)
        layer_predictions = np.hstack(oof_predictions_per_estimator)

        return layer_predictions, trained_layer_models

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions using the trained Stacking Classifier.

        :param X: Test features (NumPy array or Pandas DataFrame).
        :return: Predictions (NumPy array - probabilities from meta-model).
        """
        if not self.meta_model:
            raise RuntimeError("StackingClassifier has not been trained. Call train() first.")
        if not self.trained_models_per_layer:
            raise RuntimeError("Base models in layers have not been trained. Call train() first.")

        current_X = X # Start with the original features for the first layer

        # Get predictions from each layer sequentially
        for i, layer_trained_estimators in enumerate(self.trained_models_per_layer):
            layer_predictions_accumulator = []
            for j, (name, model) in enumerate(layer_trained_estimators): # Use the full-data trained models
                y_pred = None
                # Determine if predict_proba should be used for inference
                if hasattr(model, 'predict_proba') and \
                   (isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)) or \
                    (isinstance(model, SVC) and getattr(model, 'probability', False))):
                    y_pred = model.predict_proba(current_X)
                elif hasattr(model, 'predict'):
                    y_pred = model.predict(current_X)
                else:
                    raise RuntimeError(f"Trained estimator {name} in layer {i+1} does not have 'predict' or 'predict_proba' method.")
                
                # Ensure predictions are 2D for consistent concatenation
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                
                layer_predictions_accumulator.append(y_pred)
            
            # Concatenate predictions from current layer's estimators to form input for next layer
            if layer_predictions_accumulator:
                current_X = np.hstack(layer_predictions_accumulator)
            else:
                # This case implies no estimators in a layer, which should ideally be prevented by validation
                current_X = np.zeros((X.shape[0], 0))

        # Get predictions from the meta-model using the final concatenated predictions
        meta_predictions = self.meta_model.predict(current_X)
        return meta_predictions

    def get_model(self):
        """
        Returns the trained meta-model.

        :return: The trained Keras meta-model.
        """
        return self.meta_model

    def get_estimators(self):
        """
        Returns the list of estimators (models) used in the Stacking Classifier.

        :return: A list of lists of (name, model) tuples as initialized (not trained).
        """
        return self.estimators
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trains the Stacking Classifier.

        :param X: Training features (NumPy array or Pandas DataFrame).
        :param y: Training target variable (NumPy array or Pandas Series).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Ensure X is a DataFrame for .iloc slicing

        # 1. Determine num_classes FIRST, before any model building that depends on it
        if y.ndim == 1: # If y is 1D (e.g., [0, 1, 2, 0])
            self.num_classes = len(np.unique(y))
        elif y.ndim > 1 and y.shape[1] == 1: # If y is 2D but single column (e.g., [[0],[1],[2]])
            self.num_classes = len(np.unique(y.ravel()))
        else: # If y is already one-hot encoded (e.g., [[1,0,0],[0,1,0]])
            self.num_classes = y.shape[1]
        
        # 2. Build or re-build the meta-model after num_classes is definitively set
        self.meta_model = self._build_meta_model()

        # 3. One-hot encode y if specified, after num_classes is known
        if self.one_hot_encode:
            # ensure y is not already one-hot encoded for safety
            if y.ndim == 1 or (y.ndim > 1 and y.shape[1] == 1):
                y = to_categorical(y, num_classes=self.num_classes)
            # If y is already one-hot and one_hot_encode is true, do nothing
            # else: warning or error if y's shape doesn't match num_classes and one_hot_encode is true

        current_X = X # Features for the current layer

        # Train each layer
        # self.trained_models_per_layer will store the models fitted on the full dataset
        # after their OOF predictions are generated.
        self.trained_models_per_layer = [] 
        for i, layer_estimators in enumerate(self.estimators):
            print(f"Training Layer {i + 1}/{len(self.estimators)}...")
            # _train_layer returns OOF predictions (layer_predictions)
            # and base models refitted on the full dataset (re_trained_layer_models)
            layer_predictions, re_trained_layer_models = self._train_layer(current_X, y, layer_estimators, self.cv_folds)
            self.trained_models_per_layer.append(re_trained_layer_models) # Store models for prediction
            current_X = layer_predictions # OOF predictions become features for the next layer

        # Train the meta-model on the predictions from the last layer (current_X holds these)
        print("Training Meta-Model...")
        self.meta_model.fit(current_X, y, epochs=self.meta_model_params.get('epochs', 10),
                             batch_size=self.meta_model_params.get('batch_size', 32),
                             validation_split=self.meta_model_params.get('validation_split', 0.0),
                             verbose=1) # Added verbose for training feedback

    def evaluate(self, X: pd.DataFrame, y: np.ndarray):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array or Pandas DataFrame).
        :param y: Test target variable (NumPy array or Pandas Series).
        :return: Evaluation metrics (dictionary).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Ensure X is a DataFrame for .iloc slicing

        # Ensure y is in the correct format for evaluation (labels)
        if y.ndim > 1 and y.shape[1] > 1: # If y is one-hot encoded
            y_true_labels = np.argmax(y, axis=1)
        else: # If y is labels (1D or 2D with single column)
            y_true_labels = y.ravel() # Flatten to 1D array of labels

        y_pred_probs = self.predict(X)  # Get the predicted probabilities from the meta-model
        y_pred_labels = np.argmax(y_pred_probs, axis=1)  # Get predicted class labels

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels)
        
        auc_roc = None
        # roc_auc_score requires probabilities and can handle multi_class='ovr' or 'ovo'
        # if the number of classes is > 2, multi_class argument is required.
        if self.num_classes > 2:
            auc_roc = roc_auc_score(y_true_labels, y_pred_probs, multi_class='ovr')
        elif self.num_classes == 2: # Binary classification
            auc_roc = roc_auc_score(y_true_labels, y_pred_probs[:, 1]) # Probabilities for the positive class
        else:
            warnings.warn("ROC AUC score not calculated: Only applicable for binary or multi-class classification.")

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}

    def save_model(self, path: str):
        """
        Saves the entire StackingClassifier model, including base models and meta-model.

        :param path: Directory path where the model components will be saved.
                     A folder will be created at this path if it doesn't exist.
        """
        if not self.meta_model:
            raise RuntimeError("Model not trained. Cannot save an untrained model.")
        
        os.makedirs(path, exist_ok=True)
        print(f"Saving StackingClassifier to: {os.path.abspath(path)}")

        # 1. Save the Keras meta-model
        # Using .h5 extension as per your example, though .keras is the recommended format for Keras 3.
        meta_model_path = os.path.join(path, "meta_model.h5") 
        try:
            self.meta_model.save(meta_model_path)
            print(f"Keras meta-model saved to {meta_model_path}")
        except Exception as e:
            print(f"Error saving Keras meta-model to {meta_model_path}: {e}")

        # 2. Save the scikit-learn base models
        base_models_config = []
        for layer_idx, layer_estimators in enumerate(self.trained_models_per_layer):
            layer_models_info = []
            for est_idx, (name, model) in enumerate(layer_estimators):
                model_filename = f"layer{layer_idx}_estimator{est_idx}_{name}.joblib" # Using .joblib extension
                model_path = os.path.join(path, model_filename)
                try:
                    joblib.dump(model, model_path)
                    layer_models_info.append({"name": name, "model_path": model_filename})
                except Exception as e:
                    print(f"Error saving scikit-learn base model {name} (Layer {layer_idx+1}, Estimator {est_idx+1}) to {model_path}: {e}")
            base_models_config.append(layer_models_info)
        print(f"Base models saved to individual .joblib files in {path}")

        # 3. Save the StackingClassifier's internal parameters and structure
        model_config = {
            "params": self.params,
            "cv_folds": self.cv_folds,
            "one_hot_encode": self.one_hot_encode,
            "num_classes": self.num_classes, # Store num_classes
            "meta_model_relative_path": os.path.basename(meta_model_path), # Store relative path
            "base_models_config": base_models_config
        }
        config_path = os.path.join(path, "stacking_classifier_config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"StackingClassifier configuration saved to: {config_path}")
        print("StackingClassifier saved successfully!")

    @classmethod
    def load_model(cls, path: str):
        """
        Loads a StackingClassifier model from the specified directory.

        :param path: Directory path where the model components were saved.
        :return: An instance of StackingClassifier with loaded models.
        """
        config_path = os.path.join(path, "stacking_classifier_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. "
                                    f"Ensure '{path}' is the correct model directory.")
        
        print(f"Loading StackingClassifier from: {os.path.abspath(path)}")
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        # 1. Initialize StackingClassifier with loaded parameters
        # Pass params from config to __init__
        instance = cls(params=model_config.get("params", {})) 
        instance.cv_folds = model_config.get("cv_folds", 5)
        instance.one_hot_encode = model_config.get("one_hot_encode", True)
        instance.num_classes = model_config.get("num_classes") # Load num_classes

        # 2. Load the Keras meta-model
        meta_model_relative_path = model_config.get("meta_model_relative_path", "meta_model.h5") # Updated for .h5
        meta_model_path = os.path.join(path, meta_model_relative_path)
        if not os.path.exists(meta_model_path):
             raise FileNotFoundError(f"Meta-model file not found at {meta_model_path}.")
        try:
            instance.meta_model = load_model(meta_model_path)
            print(f"Meta-model loaded from: {meta_model_path}")
        except Exception as e:
            print(f"Error loading Keras meta-model from {meta_model_path}: {e}")
            raise # Re-raise the exception after logging

        # 3. Load the scikit-learn base models
        loaded_trained_models_per_layer = []
        base_models_config = model_config.get("base_models_config", [])
        for layer_info in base_models_config:
            layer_models = []
            for est_info in layer_info:
                name = est_info["name"]
                model_filename = est_info["model_path"]
                model_path = os.path.join(path, model_filename)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Base model file not found at {model_path}.")
                try:
                    model = joblib.load(model_path)
                    layer_models.append((name, model))
                except Exception as e:
                    print(f"Error loading scikit-learn base model {name} from {model_path}: {e}")
                    raise # Re-raise the exception after logging
            loaded_trained_models_per_layer.append(layer_models)
        instance.trained_models_per_layer = loaded_trained_models_per_layer
        print("Base models loaded.")
        print("StackingClassifier loaded successfully!")
        return instance

