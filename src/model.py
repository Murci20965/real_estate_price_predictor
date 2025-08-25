# src/model.py

import xgboost as xgb

# Import the model hyperparameters from our config file
from src.config import XGBOOST_PARAMS

def create_model() -> xgb.XGBRegressor:
    """
    Creates and returns an XGBoost Regressor model
    with predefined hyperparameters.

    Returns:
        xgb.XGBRegressor: The XGBoost model instance.
    """
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    return model