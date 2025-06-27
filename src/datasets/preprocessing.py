import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(X, y, test_size=0.2, random_state=None, scaling='standard', handle_missing='default'):
    """
    This function preprocesses the data by splitting, scaling, and handling missing values.

    :param X: Input features (NumPy array).
    :param y: Target variable (NumPy array).
    :param test_size: The proportion of the test set size.
    :param random_state: Random state for data splitting.
    :param scaling: Scaling method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), or None.
    :param handle_missing: Method to handle missing values: 'default', 'mean', 'median', or 'drop'.
        'default': Impute only if there are any missing values, uses mean for numerical and most frequent for categorical.
        'mean': Fills missing values with the mean of each column.
        'median': Fills missing values with the median of each column.
        'drop': Drops rows containing any missing values.
    :return: X_train, X_test, y_train, y_test (NumPy arrays).
    """
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 2. Handle missing values
    if handle_missing is not None:
        X_train, X_test = handle_missing_values(X_train, X_test, method=handle_missing)

    # 3. Scale the features
    if scaling is not None:
        X_train, X_test = scale_features(X_train, X_test, method=scaling)

    # 4. Encode the target variable (if necessary)
    y_train, y_test = encode_target_variable(y_train, y_test)

    return X_train, X_test, y_train, y_test

def handle_missing_values(X_train, X_test, method='default'):
    """
    Handle missing values in training and testing data.

    :param X_train: Training features data (NumPy array).
    :param X_test: Testing features data (NumPy array).
    :param method: Method to handle missing values: 'default', 'mean', 'median', or 'drop'.
    :return: The processed X_train, X_test.
    """
    if method == 'drop':
        X_train = X_train[~np.any(np.isnan(X_train), axis=1)]
        X_test = X_test[~np.any(np.isnan(X_test), axis=1)]
        return X_train, X_test

    elif method in ['mean', 'median', 'default']:
        if method == 'default':
            # Check if there are any missing values to impute only when necessary
            if np.any(np.isnan(X_train)):
                for i in range(X_train.shape[1]):
                    if np.issubdtype(X_train[:, i].dtype, np.number):  # Numerical values
                        imputer = SimpleImputer(missing_values=np.nan, strategy='mean' if method != 'median' else 'median')
                        X_train[:, i] = imputer.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()
                        X_test[:, i] = imputer.transform(X_test[:, i].reshape(-1, 1)).flatten()
                    else:  # Categorical values
                        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                        X_train[:, i] = imputer.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()
                        X_test[:, i] = imputer.transform(X_test[:, i].reshape(-1, 1)).flatten()
            else:
                print("No missing values in the training set. Skipping imputation.")
        else:  # 'mean' or 'median'
            imputer = SimpleImputer(missing_values=np.nan, strategy=method)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
        return X_train, X_test
    else:
        raise ValueError(f"Invalid method for handling missing values: {method}")

def scale_features(X_train, X_test, method='standard'):
    """
    Scale training and testing features.

    :param X_train: Training features data (NumPy array).
    :param X_test: Testing features data (NumPy array).
    :param method: Scaling method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), or None.
    :return: The scaled X_train, X_test.
    """
    if method == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif method is None:
        return X_train, X_test
    else:
        raise ValueError(f"Invalid scaling method: {method}")
    return X_train, X_test

def encode_target_variable(y_train, y_test):
    """
    Encode the target variable.

    :param y_train: Training target variable (NumPy array).
    :param y_test: Testing target variable (NumPy array).
    :return: The encoded y_train, y_test.
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    return y_train, y_test

if __name__ == '__main__':
    # Test the preprocessing function
    # Create dummy data for testing
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, 2, 3], [4, np.nan, 6], [7, 8, np.nan]])
    y = np.array(['a', 'b', 'a', 'b', 'a', 'b'])

    print("Original X:")
    print(X)
    print("Original y:", y)

    # Test handling missing values
    X_train, X_test, y_train, y_test = preprocess_data(X, y, handle_missing='mean')
    print("\nAfter handling missing values (mean):")
    print("X_train:", X_train)
    print("X_test:", X_test)

    X_train, X_test, y_train, y_test = preprocess_data(X, y, handle_missing='median')
    print("\nAfter handling missing values (median):")
    print("X_train:", X_train)
    print("X_test:", X_test)

    X_train, X_test, y_train, y_test = preprocess_data(X, y, handle_missing='drop')
    print("\nAfter handling missing values (drop):")
    print("X_train:", X_train)
    print("X_test:", X_test)

    # Test scaling
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaling='standard')
    print("\nAfter standard scaling:")
    print("X_train:", X_train)
    print("X_test:", X_test)

    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaling='minmax')
    print("\nAfter minmax scaling:")
    print("X_train:", X_train)
    print("X_test:", X_test)

    # Test encoding
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print("\nAfter label encoding:")
    print("y_train:", y_train)
    print("y_test:", y_test)
