# src/config.py

import pathlib
from datetime import datetime

# Define the root directory of the project
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

# --- PATHS AND FILENAMES ---

# Define paths to key directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Raw data file
RAW_DATA_FILE = RAW_DATA_DIR / "train.csv"

# --- START OF NEW CODE ---
# Processed data files
# These files will be created by the preprocessing script
PROCESSED_X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
PROCESSED_X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"
PROCESSED_Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
PROCESSED_Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"
# --- END OF NEW CODE ---

# Naming convention for the saved model files
MODEL_NAME_PREFIX = "xgboost_model"
MODEL_FILE_EXTENSION = ".joblib"

def get_versioned_model_name():
    """Generates a model filename with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME_PREFIX}_{timestamp}{MODEL_FILE_EXTENSION}"


# --- FEATURE LISTS ---

# The target variable we are trying to predict
TARGET_VARIABLE = "SalePrice"

# Features to drop from the dataset
# 'Id' is just an identifier and provides no predictive value.
FEATURES_TO_DROP = ["Id"]

# Categorical features that we will one-hot encode
CATEGORICAL_FEATURES = [
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType",
    "SaleCondition"
]

# Numerical features for which we will impute missing values
NUMERICAL_FEATURES = [
    "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGr", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
    "MoSold", "YrSold"
]




# --- MODEL CONFIGURATION ---

# Hyperparameters for the XGBoost model
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1  # Use all available CPU cores
}