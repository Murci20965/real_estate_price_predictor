# src/train.py

import joblib
import numpy as np
from src.evaluate import calculate_rmse, calculate_r2

# Import our custom modules
from src import config
from src.logger_config import logger
from src.preprocessing import run_preprocessing
from src.model import create_model

def train_model():
    """
    Main function to train the model.
    It runs the preprocessing pipeline, trains the XGBoost model,
    evaluates it, and saves the trained model and preprocessor.
    """
    logger.info("--- Starting the training pipeline ---")

    # 1. Run the preprocessing pipeline
    logger.info("Step 1/5: Running data preprocessing...")
    try:
        X_train, X_test, y_train, y_test, preprocessor = run_preprocessing()
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        return

    # 2. Create the model
    logger.info("Step 2/5: Creating the XGBoost model...")
    model = create_model()
    logger.info("Model created.")

    # 3. Train the model
    logger.info("Step 3/5: Training the model...")
    try:
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        return

    # 4. Save the trained model and the preprocessor
    logger.info("Step 4/5: Saving artifacts...")
    try:
        # Get the versioned model name
        model_filename = config.get_versioned_model_name()
        model_save_path = config.MODEL_DIR / model_filename
        joblib.dump(model, model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

        # Save the preprocessor
        preprocessor_save_path = config.MODEL_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_save_path)
        logger.info(f"Preprocessor saved to: {preprocessor_save_path}")

    except Exception as e:
        logger.error(f"An error occurred while saving artifacts: {e}")
        return

    # 5. Evaluate the model on the test set
    logger.info("Step 5/5: Evaluating the model...")
    try:
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = calculate_rmse(y_test, y_pred)
        r2 = calculate_r2(y_test, y_pred)

        logger.info(f"Evaluation Metrics on Test Set:")
        logger.info(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"  - R-squared (R²): {r2:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        return
    
    logger.info("--- Training pipeline finished successfully ---")

if __name__ == "__main__":
    train_model()