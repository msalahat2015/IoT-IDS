import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from ..base.base_algorithm import BaseAlgorithm  # type: ignore # Import the base class
import json
import inspect


class VotingClassifierModel(BaseAlgorithm):
    """
    A Voting Classifier that combines multiple models.
    """

    def __init__(self, params=None):
        """
        Initializes the Voting Classifier.

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

        self.voting = self.params.get('voting', {}).get('method', 'hard')
        self.weights = self.params.get('voting', {}).get('weights', None)
        self.estimators = self._initialize_estimators()
        self.model = self._build_model()

    def _initialize_estimators(self):
        """
        Initializes the estimators based on the parameters in self.params.
        :return: A list of (name, model) tuples.
        """
        estimators_param = self.params.get('estimators', [])  # Get estimators from voting
        if not estimators_param:
            raise ValueError("No estimators provided in parameters.")

        estimators = []
        for est_param in estimators_param:
            name = est_param.get('name')
            model_name = est_param.get('model')
            model_params = {
                k: v
                for k, v in est_param.items()
                if k not in ['name', 'model']
            }  # Extract model parameters

            if not name:
                raise ValueError("Each estimator must have a 'name'.")
            if not model_name:
                raise ValueError("Each estimator must have a 'model' class path.")

            # Dynamically import the model class
            try:
                module_name, class_name = model_name.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                # Check for invalid parameters
                valid_params = list(inspect.signature(model_class).parameters.keys())
                invalid_params = [
                    k for k in model_params if k not in valid_params
                ]
                if invalid_params:
                    raise ValueError(
                        f"Invalid parameters {invalid_params} for model {model_name}: {invalid_params}"
                    )
                model = model_class(**model_params)  # Instantiate with parameters
            except ImportError as e:
                raise ValueError(
                    f"Could not import model {model_name}.  Check the module path: {e}"
                )
            except AttributeError as e:
                raise ValueError(
                    f"Could not find class {class_name} in module {module_name}: {e}"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not initialize model {model_name}: {e}"
                )

            estimators.append((name, model))
        return estimators

    def _build_model(self):
        """
        Builds the VotingClassifier model.
        :return: A VotingClassifier model.
        """
        if not self.estimators:
            raise ValueError("Estimators must be provided in the params dictionary.")
        if self.voting not in ['hard', 'soft']:
            raise ValueError(
                f"Invalid voting method: {self.voting}.  Must be 'hard' or 'soft'."
            )
        model = VotingClassifier(
            estimators=self.estimators,
            voting=self.voting,  # Use the value from self.voting
            weights=self.weights,
        )
        return model

    def train(self, X, y):
        """
        Trains the Voting Classifier.

        :param X: Training features (NumPy array).
        :param y: Training target variable (NumPy array).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Makes predictions.

        :param X: Test features (NumPy array).
        :return: Predictions (NumPy array).
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the model's performance.

        :param X: Test features (NumPy array).
        :param y: Test target variable (NumPy array).
        :return: Evaluation metrics (dictionary).
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        auc_roc = 0  # Initialize auc_roc

        # Calculate ROC AUC.
        if self.voting == 'soft':
            try:
                y_pred_proba = self.model.predict_proba(X)
                if len(y_pred_proba[0]) == 2:
                    auc_roc = roc_auc_score(y, y_pred_proba[:, 1])
                else:
                    le = LabelEncoder()
                    y_true = le.fit_transform(y)
                    auc_roc = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr'
                    )
            except Exception as e:
                print(
                    f"Error calculating ROC AUC: {e}.  Returning 0 for AUC.  Check model probabilities and data."
                )

        return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}

    def get_model(self):
        """
        Returns the trained model.

        :return: The trained model
        """
        return self.model

    def get_estimators(self):
        """
        Returns the list of estimators (models) used in the Voting Classifier.

        :return: A list of (name, model) tuples.
        """
        return self.estimators


## 1st try


# import numpy as np
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from ..base.base_algorithm import BaseAlgorithm  # type: ignore
# from sklearn.preprocessing import LabelEncoder #For multiclass
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

# class VotingClassifierModel(BaseAlgorithm):
#     """
#     A Voting Classifier that combines multiple models.
#     """
#     def __init__(self, params=None):
#         """
#         Initializes the Voting Classifier.

#         :param params: Algorithm parameters (dictionary).  Must include 'estimators'.
#         """
#         if params is None:
#             params = {}
#         self.params = params
#         self.estimators = params.get('estimators', [])  # List of (name, model) tuples
#         self.voting = params.get('voting', 'hard') # 'hard' or 'soft'
#         self.weights = params.get('weights', None) # weights for soft voting
#         self.model = self._build_model()

#     def _build_model(self):
#         """
#         Builds the VotingClassifier model.
#         :return: A VotingClassifier model.
#         """
#         if not self.estimators:
#             raise ValueError("Estimators must be provided in the params dictionary.")

#         model = VotingClassifier(estimators=self.estimators, voting=self.voting, weights=self.weights)
#         return model

#     def train(self, X, y):
#         """
#         Trains the Voting Classifier.

#         :param X: Training features (NumPy array).
#         :param y: Training target variable (NumPy array).
#         """
#         self.model.fit(X, y)

#     def predict(self, X):
#         """
#         Makes predictions.

#         :param X: Test features (NumPy array).
#         :return: Predictions (NumPy array).
#         """
#         return self.model.predict(X)

#     def evaluate(self, X, y):
#         """
#         Evaluates the model's performance.

#         :param X: Test features (NumPy array).
#         :param y: Test target variable (NumPy array).
#         :return: Evaluation metrics (dictionary).
#         """
#         y_pred = self.predict(X)
#         accuracy = accuracy_score(y, y_pred)
#         report = classification_report(y, y_pred)

#         # Calculate ROC AUC.  Handles binary and multiclass cases.
#         if self.voting == 'soft':
#             try:
#                 y_pred_proba = self.model.predict_proba(X)
#                 # Check if it is binary or multiclass
#                 if len(y_pred_proba[0]) == 2:
#                   auc_roc = roc_auc_score(y, y_pred_proba[:, 1])
#                 else:
#                   le = LabelEncoder()
#                   y_true = le.fit_transform(y)
#                   auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
#             except:
#                 auc_roc = 0
#         else:
#             auc_roc = 0 #Hard voting doesn't provide probabilities

#         return {"accuracy": accuracy, "classification_report": report, "auc_roc": auc_roc}
    
#     def get_model(self):
#         """
#         Returns the trained model.
#         :return: The trained model
#         """
#         return self.model

#     def get_estimators(self):
#         """
#         Returns the list of estimators (models) used in the Voting Classifier.
        
#         :return: A list of (name, model) tuples.
#         """
#         # Proposed models:
#         model1 = LogisticRegression(random_state=1)
#         model2 = DecisionTreeClassifier(random_state=1)
#         model3 = SVC(random_state=1, probability=True) # probability=True for soft voting

#         estimators = [
#             ('lr', model1),
#             ('dt', model2),
#             ('svc', model3)
#         ]
#         return estimators