# app/predict.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Import our custom modules
from src.config import MODEL_DIR
from src.logger_config import logger
from src.preprocessing import engineer_features

def load_latest_model():
    """
    Loads the most recently trained model and the preprocessor.

    Returns:
        tuple: A tuple containing the loaded model and preprocessor.
    """
    try:
        # Find the most recent model file in the model directory
        model_files = list(Path(MODEL_DIR).glob("*.joblib"))
        if not model_files:
            logger.error("No model files found in the directory.")
            return None, None
        
        # Filter out the preprocessor file before finding the latest model
        model_files = [f for f in model_files if "preprocessor" not in f.name]
        if not model_files:
            logger.error("No model files (excluding preprocessor) found.")
            return None, None

        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        model = joblib.load(latest_model_path)
        logger.info(f"Loaded latest model: {latest_model_path.name}")

        # Load the preprocessor
        preprocessor_path = Path(MODEL_DIR) / "preprocessor.joblib"
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Loaded preprocessor.")
        
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return None, None

def make_prediction(input_data: dict, model, preprocessor) -> float:
    """
    Makes a price prediction on a single instance of input data.

    Args:
        input_data (dict): A dictionary containing the features for one house.
        model: The trained machine learning model.
        preprocessor: The fitted preprocessing pipeline.

    Returns:
        float: The predicted house price.
    """
    try:
        # Convert the input dictionary to a pandas DataFrame
        df = pd.DataFrame([input_data])
        
        # Manually calculate 'TotalBsmtSF' as it's not in the API input model
        df['TotalBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']
        
        # Apply the same feature engineering as in training
        df_engineered = engineer_features(df)

        # Preprocess the data using the loaded preprocessor
        processed_data = preprocessor.transform(df_engineered)
        
        # Make a prediction on the log-transformed scale
        log_prediction = model.predict(processed_data)
        
        # Invert the log transformation to get the actual price
        prediction = np.expm1(log_prediction[0])
        
        # --- START OF FIX ---
        # Convert the numpy float to a standard Python float
        return float(prediction)
        # --- END OF FIX ---

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return None
