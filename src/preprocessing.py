# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import configuration variables from our config file
from src import config

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features based on existing ones.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The data with new features added.
    """
    # Create TotalSF by summing up the square footage of basement, 1st, and 2nd floors
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    # Create HouseAge from the difference between the year sold and year built
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    # Create a binary feature indicating if a remodel occurred
    data['WasRemodeled'] = (data['YearRemodAdd'] != data['YearBuilt']).astype(int)

    # Drop the original columns that are now redundant or were used to create new features
    # This helps to reduce multicollinearity and model complexity.
    data = data.drop(['YrSold', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
    
    return data

def create_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a scikit-learn preprocessing pipeline.

    The pipeline will:
    1. Impute missing values for numerical features with the median.
    2. Scale numerical features.
    3. Impute missing values for categorical features with the string 'None'.
    4. One-hot encode categorical features.

    Args:
        numerical_features (list): List of numerical column names.
        categorical_features (list): List of categorical column names.

    Returns:
        ColumnTransformer: The scikit-learn preprocessing pipeline.
    """
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor

def run_preprocessing():
    """
    Executes the full preprocessing pipeline and saves the processed data.
    """
    data = load_data(config.RAW_DATA_FILE)

    X = data.drop(columns=[config.TARGET_VARIABLE] + config.FEATURES_TO_DROP)
    y = data[config.TARGET_VARIABLE]
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # --- START OF NEW CODE ---
    # Save the engineered and split dataframes for inspection and debugging
    X_train.to_csv(config.PROCESSED_X_TRAIN_FILE, index=False)
    X_test.to_csv(config.PROCESSED_X_TEST_FILE, index=False)
    y_train.to_csv(config.PROCESSED_Y_TRAIN_FILE, index=False)
    y_test.to_csv(config.PROCESSED_Y_TEST_FILE, index=False)
    # --- END OF NEW CODE ---

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = create_preprocessor(numerical_features, categorical_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing complete and intermediate files saved.")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
