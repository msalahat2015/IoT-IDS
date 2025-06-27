# main.py
import datetime
from datetime import datetime as dt

import sys
import os

import numpy as np
import pandas as pd
import joblib # Import joblib for saving scikit-learn models
from tensorflow.keras.models import save_model # Import for saving Keras models
from tensorflow.keras.models import load_model as keras_load_model # Import for loading Keras models


# Add the path to src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets import data_loader  # Import the data loading function
from src.datasets.preprocessing import preprocess_data
from src.datasets.custom.cic_preprocessing import preprocess_data


from src.evaluation.metrics import calculate_metrics  # Import the evaluation metrics function
import argparse  # Import the argparse module for command-line arguments
import json  # Import json

from src.algorithms.machine_learning.random_forest import RandomForest
from src.algorithms.machine_learning.xgboost import XGBoostClassifier
from src.algorithms.machine_learning.decision_tree import DecisionTree
from src.algorithms.machine_learning.gradient_boosting import GradientBoosting
from src.algorithms.machine_learning.knn import KNN  # Import KNN
from src.algorithms.machine_learning.naive_bayes import NaiveBayes  # Import NaiveBayes
from src.algorithms.machine_learning.svm import SVM  # Import SVM
from src.algorithms.machine_learning.extra_trees_classifier import ExtraTrees  # Import ExtraTrees
from src.algorithms.machine_learning.pca import PCAWrapper  # Import PCAWrapper
from src.algorithms.machine_learning.balanced_svm import BalancedSVM  # Import BalancedSVM
from src.algorithms.machine_learning.weighted_svm import WeightedSVM  # Import WeightedSVM
from src.algorithms.machine_learning.logistic_regression import LogisticRegressionClassifier
from src.algorithms.machine_learning.lightgbm import LightGBM
from src.algorithms.machine_learning.adaboost import AdaBoost
from src.algorithms.machine_learning.isolation_forest import IsolationForestClass
from src.algorithms.machine_learning.localoutlierclass_lof import LOFClass
from src.algorithms.machine_learning.perceptron import PerceptronClassifier  # Import PerceptronClassifier
from src.algorithms.machine_learning.threshold import ThresholdClassifier  # Import ThresholdClassifier
from src.algorithms.machine_learning.autoencoder import Autoencoder  # Import Autoencoder
from src.algorithms.machine_learning.mlp import MLP  # Import Autoencoder

from src.algorithms.deep_learning.ann import ANN  # Import
from src.algorithms.deep_learning.dnn import DNNModel  # Import
from src.algorithms.deep_learning.rnn import RNNModel  # Import
from src.algorithms.deep_learning.lstm import LSTMModel
from src.algorithms.deep_learning.cnn import CNNModel

from src.algorithms.reinforcement_learning.dqn import DQN  # Import the DQN class
from src.algorithms.custom_learning.voting import VotingClassifierModel
from src.algorithms.custom_learning.stacking import StackingClassifier


def test_model(model, X_test, algorithm_label):
    """
    Tests the trained model and measures its prediction time.

    :param model: The trained model object.
    :param X_test: The test features.
    :param algorithm_label: The label of the algorithm for specific handling (e.g., DQN).
    :return: Tuple containing (predictions, prediction_time_seconds).
    """
    y_pred = None
    prediction_time_seconds = 0.0
    start_time = dt.now()

    if algorithm_label == 'DQN':
        # For DQN, prediction usually involves getting an action for a given state
        # This is a simplified approach; a full DQN evaluation would involve an environment.
        if hasattr(model, 'act'):
            # Act on all test samples; this might not be a typical DQN evaluation
            # but serves for measuring a form of prediction time.
            y_pred_list = []
            for i in range(len(X_test)):
                sample_state = np.array([X_test[i]])
                y_pred_list.append(model.act(sample_state))
            y_pred = np.array(y_pred_list)
        else:
            print(f"Warning: DQN model does not have an 'act' method for testing.")
            y_pred = None
    elif hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
        # For Keras models wrapped in custom classes
        y_pred = model.model.predict(X_test)
    else:
        print(f"Warning: Model {algorithm_label} does not have a 'predict' method for testing.")
        y_pred = None
    
    end_time = dt.now()
    prediction_time = end_time - start_time
    prediction_time_seconds = prediction_time.total_seconds()

    return y_pred, prediction_time_seconds


def main(data_path, output_path, algorithms_config_path="algorithms.json"):
    """
    Runs an experiment using various algorithms, including DQN,
    with configuration from a JSON file and logs results to individual files
    and a summary file.

    :param data_path: Path to the data file.
    :param output_path: Path to the directory to save output files.
    :param algorithms_config_path: Path to the JSON file containing algorithm configurations.
    """
    # Debug prints to check received arguments
    print(f"DEBUG: sys.argv: {sys.argv}")
    print(f"DEBUG: Received data_path: {data_path}")
    print(f"DEBUG: Received output_path: {output_path}")
    print(f"DEBUG: Received algorithms_config_path: {algorithms_config_path}")


    # 1. Load the data
    data_load_option = 2  # 1 for X, y; 2 for DataFrame
    try:
        if data_load_option == 1:
            # Option 1: Load data into X, y
            X, y = data_loader.load_data(data_path)
            pass  # added pass
            data_size = X.shape[0]
        elif data_load_option == 2:
            # Option 2: Load data into a DataFrame
            df = data_loader.load_csv_to_dataframe(data_path)
            data_size = df.shape[0]  # get the number of rows in dataframe
            # You would then adapt the rest of your code to work with the DataFrame 'df'
            # For example, if you still need to separate it into X and y:
            # X = df.iloc[:, :-1].values
            # y = df.iloc[:, -1].values
            # But the specific way you do this depends on your application's needs.
        else:
            raise ValueError("Invalid data_load_option.  Must be 1 or 2.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)  # Exit the program if the data file is not found

    # 2. Preprocess the data
    if data_load_option == 1:
        X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.2, random_state=42)
    elif data_load_option == 2:
        X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.2, random_state=42)
        
    # 3. Load algorithms configuration from JSON
    try:
        with open(algorithms_config_path, 'r') as f:
            algorithms_config = json.load(f)
    except Exception as e:
        print(f"Error loading algorithms from {algorithms_config_path}: {e}")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)  # Create a timestamped subdirectory
    filenameAll = os.path.join(output_dir, f"All-Algorithms_{timestamp}.txt")

    # Write header for the AllResult file
    with open(filenameAll, 'w') as all_results_file:
        all_results_file.write(f"Experiment Timestamp: {timestamp}\n")
        all_results_file.write(f"Data File: {os.path.basename(data_path)}\n")
        all_results_file.write("----------------------------------------------------------------------------------------\n")
        all_results_file.write(
            "{:<30} {:<15} {:<10} {:<10} {:<10} {:<10} {:<15} \n".format(
                "Algorithm", "Train Time (s)", "Accuracy", "Precision", "Recall", "F1-Score", "Predict Time (s)"
            )
        )
        all_results_file.write("----------------------------------------------------------------------------------------\n")

    # 4. Loop through the algorithms in the configuration
    for algo_config in algorithms_config:
        algorithm_label = algo_config['label']
        algorithm_name = algo_config['algorithm']
        algorithm_params = algo_config['params']

        print(f"Running algorithm: {algorithm_label}")

        # 5. Instantiate the model
        model = None
        if algorithm_label == 'RandomForestClassifier':
            model = RandomForest(params=algorithm_params)
        elif algorithm_label == 'ANN':
            model = ANN(params=algorithm_params)
        elif algorithm_label == 'LogisticRegression':
            model = LogisticRegressionClassifier(params=algorithm_params)
        elif algorithm_label == 'DQN':
            # Assuming your data_path leads to data suitable for DQN
            # You'll likely need to adapt the state_size and action_size
            # based on your specific problem. The loaded 'X' and 'y'
            # might need further processing to fit the DQN's expected input.
            state_size = X_train.shape[1]  # Example: number of features
            # Assuming 'y' represents discrete actions (you might need to adjust)
            action_size = len(np.unique(y_train))
            model = DQN(state_size, action_size, params=algorithm_params)
        elif algorithm_label == 'XGBoostClassifier':
            model = XGBoostClassifier(params=algorithm_params)  # Create an instance of XGBoostClassifier
        elif algorithm_label == 'DecisionTreeClassifier':
            model = DecisionTree(params=algorithm_params)  # Add DecisionTreeClassifier
        elif algorithm_label == 'GradientBoostingClassifier':
            model = GradientBoosting(params=algorithm_params)  # Add GradientBoostingClassifier
        elif algorithm_label == 'GaussianNB':
            model = NaiveBayes(params=algorithm_params, nb_type='gaussian')  # GaussianNB
        elif algorithm_label == 'MultinomialNB':
            model = NaiveBayes(params=algorithm_params, nb_type='multinomial')  # MultinomialNB
        elif algorithm_label == 'SVM':
            model = SVM(params=algorithm_params)  # Instantiate SVM
        elif algorithm_label == 'ExtraTreesClassifier':
            model = ExtraTrees(params=algorithm_params)  # Instantiate ExtraTrees
        elif algorithm_label == 'PCA':
            model = PCAWrapper(params=algorithm_params)  # Instantiate PCAWrapper
        elif algorithm_label == 'BalancedSVM':
            model = BalancedSVM(params=algorithm_params)  # Instantiate BalancedSVM
        elif algorithm_label == 'WeightedSVM':
            model = WeightedSVM(params=algorithm_params)  # Instantiate WeightedSVM
        elif algorithm_label == 'Perceptron':
            model = PerceptronClassifier(params=algorithm_params)  # Instantiate PerceptronClassifier
        elif algorithm_label == 'ThresholdClassifier':
            model = ThresholdClassifier(params=algorithm_params)  # Instantiate ThresholdClassifier
        elif algorithm_label == 'KNN':
            model = KNN(params=algorithm_params)  # Instantiate ThresholdClassifier
        elif algorithm_label == 'LightGBM':
            model = LightGBM(params=algorithm_params)  # Instantiate LightGBMClassifier
        elif algorithm_label == 'Adaboost':
            model = AdaBoost(params=algorithm_params)  # Instantiate AdaboostClassifier
        elif algorithm_label == 'IsolationForestClass':
            model = IsolationForestClass(params=algorithm_params)  # Instantiate LightGBMClassifier
        elif algorithm_label == 'MLP':
            model = MLP(params=algorithm_params)  # Instantiate MLP
        elif algorithm_label == 'LOFClass':
            model = LOFClass(params=algorithm_params)  # Instantiate AdaboostClassifier
        elif algorithm_label == 'RNN':
            model = RNNModel(params=algorithm_params)
        elif algorithm_label == 'DNN':
            model = DNNModel(params=algorithm_params)
        elif algorithm_label == 'CNN':
            model = CNNModel(params=algorithm_params)
        elif algorithm_label == 'lstm':
            model = LSTMModel(params=algorithm_params)
        elif algorithm_label == 'VotingClassifierModel':
            model = VotingClassifierModel(params=algorithm_params)
        elif algorithm_label == 'StackingClassifier':
            model = StackingClassifier(params=algorithm_params)
        elif algorithm_label == 'Autoencoder':
            model = Autoencoder(params=algorithm_params)  # Instantiate Autoencoder
        else:
            print(f"Algorithm {algorithm_label} not supported. Skipping.")
            continue

        # 6. Train the model
        start_time = dt.now()
        if algorithm_label == 'DQN':
            # DQN training typically involves interacting with an environment
            # and updating the model based on experiences. This is a simplified
            # example and might need significant adaptation based on your use case.
            print("Note: DQN training requires an environment interaction loop.")
            # Placeholder for a simplified training step with the loaded data
            # This is likely NOT how you'd typically train a DQN.
            # You'd usually have a loop that interacts with an environment,
            # collects experiences, and then trains the DQN.
            if hasattr(model, 'train'):
                # Reshape data if needed for DQN input (example)
                states = np.array([X_train[i] for i in range(len(X_train))])
                actions = np.array([y_train[i] for i in range(len(y_train))])
                # Assuming some dummy rewards, next states, and done flags
                rewards = np.zeros(len(X_train))
                next_states = np.roll(states, shift=-1, axis=0)
                dones = np.zeros(len(X_train), dtype=bool)
                dones[-1] = True  # Mark the last state as terminal (example)

                for i in range(len(X_train)):
                    model.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    model.train(states[i], actions[i], rewards[i], next_states[i], dones[i])
                if hasattr(model, 'update_target_model'):
                    model.update_target_model()  # Update target network periodically
            else:
                print(f"Warning: DQN model does not have a 'train' method compatible with this data format.")
        elif model is not None and hasattr(model, 'train'):
            model.train(X_train, y_train)
        end_time = dt.now()
        training_time = end_time - start_time
        training_time_seconds = training_time.total_seconds()
        print(f"Training finished in {training_time_seconds:.4f} seconds.")

        # 7. Make predictions and measure prediction time
        y_pred, prediction_time_seconds = test_model(model, X_test, algorithm_label)

        evaluation_results = {}
        metrics = {}
        if y_pred is not None:
            if algorithm_label == 'DQN':
                print(f"DQN Predicted Action for sample state: {y_pred}")
                evaluation_results = {"note": "Evaluation of DQN requires interaction with an environment."}
                metrics = {"note": "Metrics for DQN are environment-dependent."}
            else:
                # If your data is NumPy arrays, convert to DataFrames (with proper column names)
                # This assumes feature names are not available; if they are, replace with actual names.
                # For demonstration, creating dummy column names.
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
                X_test_df = pd.DataFrame(X_test, columns=feature_names) 
                y_test_df = pd.DataFrame(y_test, columns=["Actual"])
                
                if y_pred.ndim > 1:
                   y_pred_labels = np.argmax(y_pred, axis=1)
                   y_pred_df = pd.DataFrame(y_pred_labels, columns=["Predicted"])
                else: 
                  y_pred_df = pd.DataFrame(y_pred, columns=["Predicted"])

                # Align indices to match rows
                y_test_df.index = X_test_df.index
                y_pred_df.index = X_test_df.index

                # Concatenate side by side
                combined_df = pd.concat([X_test_df, y_test_df, y_pred_df], axis=1)

                # Save to CSV
                combined_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
                
                print(f"y_pred.shape: {y_pred.shape}")
                # 8. Evaluate the model
                print("Shape of y_test:", y_test.shape)
                print("Sample of y_test:", y_test[:5])
                print("Shape of y_pred:", y_pred.shape)
                print("Sample of y_pred:", y_pred[:5])
                evaluation_results = {}
                if hasattr(model, 'evaluate'):
                    evaluation_results = model.evaluate(X_test, y_test)
                    
                # Convert predictions to the correct shape if necessary for metric calculation
                # This part needs careful handling depending on the actual output of your models
                # For classification, ensure y_pred is discrete labels.
                final_y_pred_for_metrics = y_pred_df.values.squeeze()
                
                # 9. Calculate and display metrics
                metrics = calculate_metrics(y_test_df.values.squeeze(), final_y_pred_for_metrics)
                print("Metrics:", metrics)
                print("Shape of y_test:", y_test.shape)
                print("Type of y_test:", y_test.dtype)
                print("y_test:", y_test)

                print("Shape of y_pred (after processing for metrics):", final_y_pred_for_metrics.shape)
                print("Type of y_pred (after processing for metrics):", final_y_pred_for_metrics.dtype)
                print("y_pred (after processing for metrics):", final_y_pred_for_metrics)

        # 10. Save the model
        model_save_path = None
        if model is not None:
            if algorithm_label == 'StackingClassifier':
             # For our custom StackingClassifier, call its dedicated save_model method
             # This method handles saving both Keras meta-model and scikit-learn base models
             # It creates its own directory structure within the specified path.
             stacking_model_dir = os.path.join(output_dir, "models", f"{algorithm_name}_saved_model_{timestamp}")
             try:
                 model.save_model(stacking_model_dir)
                 print(f"StackingClassifier model saved to {stacking_model_dir}")
                 model_save_path = stacking_model_dir # Set this for logging/reporting if needed
             except Exception as e:
                    print(f"Error saving StackingClassifier model {algorithm_name}: {e}")
            # Consider re-raising or handling more robustly based on your application's needs  
            if hasattr(model, 'model') and hasattr(model.model, 'save'):
                # For Keras models (ANN, DNN, RNN, CNN, LSTM, MLP, Autoencoder)
                model_save_path = os.path.join(output_dir, "models", f"{algorithm_name}_model_{timestamp}.h5")
                # Create the 'models' subdirectory if it doesn't exist
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                try:
                    model.model.save(model_save_path)
                    print(f"Keras model saved to {model_save_path}")
                except Exception as e:
                    print(f"Error saving Keras model {algorithm_name}: {e}")
            elif hasattr(model, 'model'): # For scikit-learn based models which store estimator in .model
                model_save_path = os.path.join(output_dir, "models", f"{algorithm_name}_model_{timestamp}.pkl")
                # Create the 'models' subdirectory if it doesn't exist
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                try:
                    joblib.dump(model.model, model_save_path)
                    print(f"Scikit-learn model saved to {model_save_path}")
                except Exception as e:
                    print(f"Error saving scikit-learn model {algorithm_name}: {e}")
            elif algorithm_label == 'DQN':
                # DQN model saving might be specific to its implementation
                # If your DQN class has a save method, call it here.
                # Example: if hasattr(model, 'save_model'): model.save_model(model_save_path)
                print("DQN model saving logic needs to be implemented within the DQN class if desired.")
            else:
                print(f"No standard saving method found for {algorithm_name} model.")


        # 11. Log results Save results to a text file
        filename = os.path.join(output_dir, f"{algorithm_name}_{timestamp}.txt")
        with open(filename, 'w') as f:
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Training Time: {training_time}\n")
            f.write(f"Prediction Time: {prediction_time_seconds:.10f} seconds\n") # Added prediction time
            f.write(f"Data Size: {data_size}\n")
            f.write("Parameters:\n")
            for key, value in algorithm_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nEvaluation Results:\n")
            for key, value in evaluation_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nMetrics:\n")
            if metrics:
                with open(filenameAll, 'a') as all_results_file:
                   all_results_file.write("{:<30} {:<15.6f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.10f} \n".format(algorithm_name, training_time_seconds, metrics["accuracy"]* 100, metrics["precision"]* 100, metrics["recall"]* 100, metrics["f1_score"]* 100, prediction_time_seconds) ) # Added prediction time
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write("  Not applicable or error occurred.\n")
            
            if model_save_path:
                f.write(f"\nModel saved to: {model_save_path}\n")

        print(f"Results saved to {filename}")

    print(f"All algorithm results saved to {filenameAll}")


if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run experiment with various algorithms.")
    parser.add_argument("data_path", type=str, help="Path to the data file.")
    parser.add_argument("output_path", type=str, help="Path to the directory to save output files.")
    parser.add_argument("--algorithms_config", type=str, default="algorithms.json",
                         help="Path to the algorithms configuration JSON file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (fraction).")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for data splitting.")

    args = parser.parse_args()  # Parse the command-line arguments

    # Call the main function with the parsed arguments
    main(args.data_path, args.output_path, args.algorithms_config)
