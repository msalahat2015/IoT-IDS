# cic_preprocessing.py
import sys
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from src.datasets.preprocessing import preprocess_data as base_preprocess_data
from src.datasets.preprocessing import handle_missing_values as base_handle_missing_values
from src.datasets.preprocessing import scale_features as base_scale_features
from src.datasets.preprocessing import encode_target_variable as base_encode_target_variable
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

def label_mapping_function(dataset, label_col, new_label_col, unknown_label='Unknown'):
    """
    Groups the original labels in the dataset into broader categories
    and adds a new column with these grouped labels to the dataset.
    Assigns a default label for any unknown old labels.

    :param dataset: Pandas DataFrame containing the original labels.
    :param label_col: Name of the column containing the original labels.
    :param new_label_col: Name of the new column to be created with grouped labels.
    :param unknown_label: The default label to assign to any original label not found in the mapping.
    :return: Pandas DataFrame with the new grouped label column.
    """
    label_mapping = {
        # DDoS
        'DDoS-ACK_Fragmentation': 'DDoS',
        'DDoS-UDP_Flood': 'DDoS',
        'DDoS-SlowLoris': 'DDoS',
        'DDoS-ICMP_Flood': 'DDoS',
        'DDoS-RSTFINFlood': 'DDoS',
        'DDoS-PSHACK_Flood': 'DDoS',
        'DDoS-HTTP_Flood': 'DDoS',
        'DDoS-UDP_Fragmentation': 'DDoS',
        'DDoS-TCP_Flood': 'DDoS',
        'DDoS-SYN_Flood': 'DDoS',
        'DDoS-SynonymousIP_Flood': 'DDoS',

        # Brute Force
        'DictionaryBruteForce': 'Brute Force',

        # Spoofing
        'MITM-ArpSpoofing': 'Spoofing',
        'DNS_Spoofing': 'Spoofing',

        # DoS
        'DoS-TCP_Flood': 'DoS',
        'DoS-HTTP_Flood': 'DoS',
        'DoS-SYN_Flood': 'DoS',
        'DoS-UDP_Flood': 'DoS',

        # Recon
        'Recon-PingSweep': 'Recon',
        'Recon-OSScan': 'Recon',
        'VulnerabilityScan': 'Recon',
        'Recon-PortScan': 'Recon',
        'Recon-HostDiscovery': 'Recon',

        # Web-based
        'SqlInjection': 'Web-based',
        'CommandInjection': 'Web-based',
        'Backdoor_Malware': 'Web-based',
        'Uploading_Attack': 'Web-based',
        'XSS': 'Web-based',
        'BrowserHijacking': 'Web-based',

        # Mirai
        'Mirai-greip_flood': 'Mirai',
        'Mirai-greeth_flood': 'Mirai',
        'Mirai-udpplain': 'Mirai',

        # Benign Traffic
        'BenignTraffic': 'Benign'
    }
    dataset[new_label_col] = dataset[label_col].map(label_mapping).fillna(unknown_label)
    return dataset
def cic_handle_specific_missing(df):
    # Example: Fill specific columns with a constant
    df['specific_feature'].fillna(0, inplace=True)
    return df

def cic_feature_engineering(df):
    # Example: Create a new feature
    df['duration_times_bytes'] = df['duration'] * df['bytes']
    return df

def cic_encode_categorical(X_train, X_test, categorical_cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
    X_test_encoded = ohe.transform(X_test[categorical_cols])

    # Create new DataFrames with encoded features
    X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, index=X_test.index)

    # Drop original categorical columns and concatenate encoded ones
    X_train = X_train.drop(categorical_cols, axis=1)
    X_train = pd.concat([X_train, X_train_encoded_df], axis=1)

    X_test = X_test.drop(categorical_cols, axis=1)
    X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

    return X_train, X_test


def preprocess_data(df,  test_size=0.2, random_state=None, scaling='standard', handle_missing='default', custom_steps=True,label_col='label', new_label_col='grouped_label', unknown_label='Unknown'):
  """
    Preprocesses the input DataFrame, including splitting into training and testing sets,
    handling missing values, encoding categorical features, and scaling.

    :param df: Input Pandas DataFrame containing both features and the target variable.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before splitting.
    :param scaling: The scaling method to use ('standard', 'minmax', or None).
    :param handle_missing: The method to handle missing values ('default', 'mean', 'median', or 'drop').
    :param custom_steps: If True, apply custom CIC preprocessing steps.
    :param label_col: Name of the column containing the original labels.
    :param new_label_col: Name of the new column for grouped labels.
    :param unknown_label: The default label to assign to any original label not found in the mapping.
    :return: A tuple containing the preprocessed training and testing data:
            (X_train, X_test, y_train, y_test).  All are NumPy arrays.
    """
  print(df.info())
  print(df.head())
  print(df.describe())
    # Apply label mapping function.
  processed_df = label_mapping_function(df.copy(), 'label', 'grouped_label','Unknown')
  
  # Verify the new label distribution
  print(processed_df['grouped_label'].value_counts())

  # Convert to DataFrame
  df_labels = pd.DataFrame(list(processed_df['grouped_label'].value_counts().items()), columns=["Class", "Count"])

  # Calculate imbalance ratios
  df_labels["Imbalance Ratio"] = df_labels["Count"].max() / df_labels["Count"]
  print(df_labels)

  # Detect columns with outliers
  outlier_columns_summary = detect_outlier_columns(df.copy(), threshold=10)
  print(outlier_columns_summary)
  
  # Columns to preprocess
  columns_to_preprocess = outlier_columns_summary['Column']

  # Apply preprocessing
  processed_df = preprocess_columns(processed_df.copy() , columns_to_preprocess)
  print(processed_df)

  
  # Apply the function to processed_df
  correlation_matrix, correlated_features = check_correlated_features_numeric_only(processed_df, threshold=0.9)

  # Ensure the correlation matrix is valid before plotting
  if correlation_matrix.isnull().values.any():
    print("Warning: Correlation matrix contains NaN values. Check for constant or non-numeric columns.")

  # Display the correlation matrix heatmap
#   plt.figure(figsize=(20, 20))
#   sys.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#   plt.title("Correlation Matrix (Numeric Columns Only)")
#   plt.show()

  # Show correlated feature pairs
  print(correlated_features)

  # Apply the feature removal plan
  processed_df_reduced = feature_removal_plan(processed_df)

  # Show the updated DataFrame columns
  print(processed_df_reduced.columns)

  # Apply the data type optimization
  optimized_df = fix_data_type_inefficiencies(processed_df_reduced)

  # Show updated data types
  print(optimized_df.dtypes)
  
  # Columns to check for suspicious zeros
  columns_to_check = ["flow_duration", "Drate", "Rate"]

  # Apply the function to fix suspicious zero values
  processed_df_with_zeros_fixed, zero_summary = fix_suspicious_zero_values(optimized_df, columns_to_check)

  # Show the results
  print(zero_summary)

  # Apply feature scaling (default: StandardScaler)
  scaled_df = scale_features(optimized_df , method="standard")

  # Show the scaled dataset
  print(scaled_df.head())

  # Detect potential leaky features
  leaky_features = detect_leaky_features(scaled_df, threshold=0.9)
 
  print(leaky_features)
  # Remove leaky features from the dataset
  processed_df_no_leakage = scaled_df.drop(columns=leaky_features, errors='ignore')

  # Show the features removed due to potential leakage
  print(leaky_features)
  print(processed_df_no_leakage.columns)
   # Remove leaky features from the dataset
  processed_df_final = remove_leaky_features(processed_df_no_leakage, leaky_features)
  
  print(processed_df_final.head())
  # List of identified leaky features to remove
  leaky_features = ['Max', 'Std', 'Min', 'Covariance', 'ack_count', 'Variance', 'fin_count', 'AVG']

  # Display the remaining columns after removal
  print(processed_df_final.columns)

  # Apply the cleaning function
  processed_df_cleaned = clean_and_prepare_data(processed_df_final)

  # Verify that NaN values are removed and 'label' column is dropped
  print(processed_df_cleaned.isnull().sum())
  print(processed_df_cleaned.columns)
  
  print(processed_df_cleaned['grouped_label'].value_counts())
  
  # Define classes to drop
  classes_to_drop = ["Web-based", "Brute Force"  , "Recon" , "Spoofing" , "Mirai" ]  # Example: Dropping very small classes

  # Drop rows where 'grouped_label' is in the list
  processed_df_filtered = processed_df_cleaned[~processed_df_cleaned['grouped_label'].isin(classes_to_drop)]
  
  # Define balancing parameters
  target_col = "grouped_label"
  sample_size = 5000  # Target number of samples per class
  random_state = 42

  # Separate features and target
  X = processed_df_filtered.drop(columns=[target_col])
  y = processed_df_filtered[target_col]

  # Define balancing parameters
  sample_size = 5600  # Target number of samples per class
  test_size = 0.1
  random_state = 42
  target_col = "grouped_label"

  # Identify categorical columns
  categorical_columns = X.select_dtypes(include=['object', 'category']).columns

  # Convert categorical features to numeric (Label Encoding for XGBoost compatibility)
  label_encoders = {}
  for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoders

  # Convert target labels to numeric values
  target_encoder = LabelEncoder()
  y_encoded = target_encoder.fit_transform(y)

  
  # Apply balancing
  X_balanced, y_balanced = balance_dataset(X, y_encoded, sample_size, random_state)

  # Convert labels back to original names
  y_balanced = target_encoder.inverse_transform(y_balanced)

  # Convert to DataFrame
  balanced_df = pd.DataFrame(X_balanced)
  balanced_df[target_col] = y_balanced
  print(balanced_df[target_col].value_counts())
  # Split features and target
  X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size, random_state=random_state, stratify=y_balanced)

  tocsv( X_train, X_test, y_train, y_test,X_balanced)

  # Encode target labels to numeric values
  y_train_encoded = target_encoder.fit_transform(y_train)
  y_test_encoded = target_encoder.transform(y_test)

  return X_train, X_test, y_train_encoded, y_test_encoded

def tocsv(X_train_raw,X_test_raw,y_train_raw,y_test_raw,X_balanced):

# Explicitly convert X_train, X_test to Pandas DataFrames and
# y_train, y_test to Pandas Series objects to ensure all methods are available.
 X_train = pd.DataFrame(X_train_raw, columns=X_balanced.columns)
 X_test = pd.DataFrame(X_test_raw, columns=X_balanced.columns)
 y_train = pd.Series(y_train_raw, name='target_column')
 y_test = pd.Series(y_test_raw, name='target_column')


 print("\nX_train shape:", X_train.shape)
 print("X_test shape:", X_test.shape)
 print("y_train shape:", y_train.shape)
 print("y_test shape:", y_test.shape)

 print("\ny_train value counts:\n", y_train.value_counts())
 print("\ny_test value counts:\n", y_test.value_counts())

# --- 4. Combine Features and Target for Saving ---
# It's good practice to save the features (X) and the target (y) together
# in the same CSV file for each set.

# Reset index to avoid issues with non-contiguous indices after splitting
# and to ensure a fresh index in the CSV.
# X_train and X_test are now guaranteed to be DataFrames.
 X_train_reset = X_train.reset_index(drop=True)
 y_train_reset = y_train.reset_index(drop=True) # y_train is guaranteed a Series
 X_test_reset = X_test.reset_index(drop=True)
 y_test_reset = y_test.reset_index(drop=True) # y_test is guaranteed a Series

# Create training DataFrame by concatenating features and the target.
# .to_frame(name='target') is now safely called on a Series.
 train_df = pd.concat([X_train_reset, y_train_reset.to_frame(name='target')], axis=1)

# Create testing DataFrame similarly.
 test_df = pd.concat([X_test_reset, y_test_reset.to_frame(name='target')], axis=1)

# --- 5. Save to CSV Files ---
 train_file_name = 'train.csv'
 test_file_name = 'test.csv'

 train_df.to_csv(train_file_name, index=False)
 test_df.to_csv(test_file_name, index=False)
# Function to detect columns with potential outliers
def detect_outlier_columns(df, threshold=10):
    """
    Detect columns where the gap between the max value and the 75th percentile is unusually large.
    Threshold indicates how many times larger the max value must be compared to the 75th percentile.
    """
    outlier_columns = []
    for column in df.select_dtypes(include=[np.number]).columns:
        max_value = df[column].max()
        percentile_75 = df[column].quantile(0.75)
        if percentile_75 > 0 and (max_value / percentile_75) > threshold:
            outlier_columns.append({
                "Column": column,
                "Max Value": max_value,
                "75th Percentile": percentile_75,
                "Gap": max_value - percentile_75,
                "Max/75th Percentile Ratio": max_value / percentile_75
            })
    return pd.DataFrame(outlier_columns)

def preprocess_columns(df, columns, log_transform=True, clip_percentile=99):
    """
    Preprocess specified columns in a DataFrame to handle outliers and skewness.
    
    Parameters:
    - df: DataFrame
    - columns: List of column names to preprocess
    - log_transform: Apply log transformation (default: True)
    - clip_percentile: Clip values above this percentile (default: 99)
    
    Returns:
    - Processed DataFrame with selected columns preprocessed
    """
    processed_df = df.copy()
    scaler = RobustScaler()
    
    for column in columns:
        # Log Transformation
        if log_transform:
            processed_df[column] = np.log1p(processed_df[column].clip(lower=0))  # Ensure no negative values
        
        # Clipping extreme values above the specified percentile
        upper_limit = processed_df[column].quantile(clip_percentile / 100.0)
        processed_df[column] = processed_df[column].clip(upper=upper_limit)
    
    # Apply RobustScaler for scaling
    processed_df[columns] = scaler.fit_transform(processed_df[columns])
    
    return processed_df

def check_correlated_features_numeric_only(df, threshold=0.9):
    """
    Check for highly correlated features in numeric columns of the dataset.

    Parameters:
    - df: DataFrame
    - threshold: Correlation coefficient threshold (default: 0.9)

    Returns:
    - correlation_matrix: Correlation matrix of numeric features
    - correlated_features: List of feature pairs with correlations above the threshold
    """
    # Select only numeric columns and drop constant columns (all identical values)
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

    # Handle NaN and infinite values
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    numeric_df.fillna(0, inplace=True)  # Replace NaNs with 0 to avoid issues

    correlation_matrix = numeric_df.corr()

    correlated_features = []
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row and abs(correlation_matrix.loc[row, col]) > threshold:
                correlated_features.append((row, col, correlation_matrix.loc[row, col]))

    # Remove duplicates by ensuring (A, B) and (B, A) are not both included
    correlated_features = list(set([tuple(sorted(pair[:2])) + (pair[2],) for pair in correlated_features]))
    correlated_features = sorted(correlated_features, key=lambda x: -abs(x[2]))

    return correlation_matrix, correlated_features

def feature_removal_plan(df):
    """
    Removes redundant features based on the provided plan.

    Parameters:
    - df: DataFrame with original features.

    Returns:
    - DataFrame with redundant features removed.
    """
    # List of redundant features to drop based on the plan
    features_to_drop = [
        "Srate",         # Keep Rate
        "LLC",           # Keep IPv
        "Weight",        # Keep Number
        "Magnitue",      # Keep AVG
        "Radius",        # Keep Std
        "IAT",           # Keep Number
        "Tot sum",       # Keep AVG
        "Tot size",      # Keep AVG
        "fin_flag_number"  # Keep ack_count
    ]
    
    # Drop the features
    processed_df = df.drop(columns=features_to_drop, errors='ignore')
    return processed_df
def fix_data_type_inefficiencies(df):
    """
    Fixes data type inefficiencies in the DataFrame.

    Parameters:
    - df: DataFrame with inefficient data types.

    Returns:
    - Optimized DataFrame with updated data types.
    """
    processed_df = df.copy()

    # Convert boolean-like flags (stored as float64) to uint8
    boolean_flag_columns = [col for col in processed_df.columns if "_flag" in col or "_count" in col]
    for col in boolean_flag_columns:
        if processed_df[col].dtype == 'float64':
            processed_df[col] = processed_df[col].astype('uint8')

    # Convert protocol/service indicator columns to categorical
    protocol_columns = ["HTTP", "HTTPS", "DNS", "TCP", "UDP", "ARP", "ICMP", "IPv", "LLC"]
    for col in protocol_columns:
        if col in processed_df.columns and processed_df[col].dtype in ['float64', 'int64']:
            processed_df[col] = processed_df[col].astype('category')

    return processed_df

def fix_suspicious_zero_values(df, columns_to_check, zero_threshold=0.5):
    """
    Fixes suspicious zero values in specified columns by investigating their presence and imputing or filtering if necessary.

    Parameters:
    - df: DataFrame with data to analyze and fix.
    - columns_to_check: List of column names to inspect for excessive zero values.
    - zero_threshold: Proportion threshold (default: 0.5). If more than this proportion of a column is zero, it's flagged for review.

    Returns:
    - DataFrame with zero-related issues addressed.
    - Summary of columns with high zero proportions.
    """
    processed_df = df.copy()
    zero_summary = []

    for column in columns_to_check:
        if column in processed_df.columns:
            # Calculate the proportion of zeros
            zero_proportion = (processed_df[column] == 0).mean()

            # If the proportion exceeds the threshold, flag it
            if zero_proportion > zero_threshold:
                zero_summary.append({
                    "Column": column,
                    "Zero Proportion": zero_proportion
                })

                # Investigate and handle zeros
                # Here, replacing zeros with the mean of non-zero values as an example
                non_zero_mean = processed_df.loc[processed_df[column] != 0, column].mean()
                processed_df[column] = processed_df[column].replace(0, non_zero_mean)

    # Create a summary DataFrame
    zero_summary_df = pd.DataFrame(zero_summary)

    return processed_df, zero_summary_df
def scale_features(df, method="standard"):
    """
    Scales numerical features to address feature scaling issues.
    
    Parameters:
    - df: DataFrame with features to scale.
    - method: "standard" (StandardScaler) or "minmax" (MinMaxScaler).
    
    Returns:
    - Scaled DataFrame.
    """
    processed_df = df.copy()
    
    # Select only numeric columns for scaling
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    
    if method == "standard":
        scaler = StandardScaler()  # Standardization (zero mean, unit variance)
    elif method == "minmax":
        scaler = MinMaxScaler()  # Normalization (0 to 1 range)
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
    
    # Apply scaling
    processed_df[numeric_columns] = scaler.fit_transform(processed_df[numeric_columns])
    
    return processed_df

def detect_leaky_features(df, threshold=0.9):
    """
    Detects potential data leakage by identifying features that are highly correlated 
    (above a specified threshold), suggesting redundancy.

    Parameters:
    - df: DataFrame containing features.
    - threshold: Correlation coefficient threshold to identify leakage (default: 0.9).

    Returns:
    - List of potentially leaky features.
    """
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Identify highly correlated features (potential leakage)
    leaky_features = set()
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row and abs(correlation_matrix.loc[row, col]) > threshold:
                leaky_features.add(row)  # Add feature that is redundant

    return list(leaky_features)
def remove_leaky_features(df, leaky_features):
    """
    Removes features identified as potential data leakage sources.

    Parameters:
    - df: DataFrame containing all features.
    - leaky_features: List of feature names to be removed.

    Returns:
    - DataFrame with leaky features removed.
    """
    return df.drop(columns=leaky_features, errors='ignore')
def clean_and_prepare_data(df):
    """
    Cleans the dataset by handling NaN values and removing the 'label' column.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - Cleaned DataFrame with NaN values removed and 'label' column dropped.
    """
    processed_df = df.copy()

    # Drop rows with NaN values
    processed_df.dropna(inplace=True)

    # Drop the 'label' column if it exists
    processed_df.drop(columns=["label"], errors="ignore", inplace=True)

    return processed_df
# Balance dataset: Undersample majority & oversample minority
def balance_dataset(X, y, sample_size, random_state):
    """
    Balances the dataset by:
    - Undersampling majority classes
    - Oversampling minority classes (SMOTE for very small ones)
    """
    # Count samples per class
    class_counts = pd.Series(y).value_counts()

    # Splitting classes into major and minor
    major_classes = class_counts[class_counts > sample_size].index
    minor_classes = class_counts[class_counts < sample_size].index

    # Undersample majority classes
    rus = RandomUnderSampler(sampling_strategy={cls: sample_size for cls in major_classes}, random_state=random_state)
    X_rus, y_rus = rus.fit_resample(X, y)

    # Oversample minority classes
    ros = RandomOverSampler(sampling_strategy={cls: sample_size for cls in minor_classes}, random_state=random_state)
    X_ros, y_ros = ros.fit_resample(X_rus, y_rus)

    # Use SMOTE for extremely small classes (less than 500 samples)
    smote_classes = {cls: sample_size for cls in minor_classes if class_counts[cls] < 500}
    if smote_classes:
        smote = SMOTE(sampling_strategy=smote_classes, random_state=random_state)
        X_final, y_final = smote.fit_resample(X_ros, y_ros)
    else:
        X_final, y_final = X_ros, y_ros

    return X_final, y_final

