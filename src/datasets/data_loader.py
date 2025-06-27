import pandas as pd
import numpy as np
import os
import glob


def load_csv_to_dataframe(data_path):
    """
    Load data from a CSV file.

    :param data_path: Path to the CSV file.
    :return: Pandas DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")


def load_data(data_path):
    """
    Load data from a CSV file.

    :param data_path: Path to the CSV file.
    :return: X (features), y (target variable) as NumPy arrays.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    try:
        df = pd.read_csv(data_path)
        # Assume the last column is the target variable
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # The last column
        return X, y
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

def load_from_folder(folder_path, file_pattern="*.csv"):
    """
    Load data from multiple files in a folder.

    :param folder_path: Path to the folder.
    :param file_pattern: File pattern (e.g., "*.csv", "*.txt").
    :return: X (features), y (target variable) as concatenated NumPy arrays.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found at {folder_path}")

    data_files = glob.glob(os.path.join(folder_path, file_pattern))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {folder_path} matching pattern {file_pattern}")

    X_list = []
    y_list = []
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            continue  # Skip to the next file

    if not X_list:
        raise Exception(f"No data could be loaded from the files in {folder_path}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

if __name__ == '__main__':
    # Test the load_data function
    try:
        X, y = load_data("data/my_data.csv")  # Replace with your file path
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Sample X:", X[:5])
        print("Sample y:", y[:5])
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("An error occurred:", e)

    # Test the load_from_folder function
    try:
        X, y = load_from_folder("data/data_folder", "*.csv")  # Replace with your path and pattern
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Sample X:", X[:5])
        print("Sample y:", y[:5])
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("An error occurred:", e)
