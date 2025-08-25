# src/evaluate.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE).
    This is the metric used by the Kaggle competition.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The RMSE score.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (R²) score.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The R² score.
    """
    return r2_score(y_true, y_pred)