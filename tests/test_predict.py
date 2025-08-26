# tests/test_predict.py

import numpy as np
import pandas as pd
from app.predict import make_prediction

# Create a mock model and preprocessor for testing
class MockModel:
    def predict(self, data):
        return np.array([12.2]) # A sample log-transformed prediction

class MockPreprocessor:
    def transform(self, data):
        return data # Pass-through transform for simplicity

def test_make_prediction():
    """
    Tests the make_prediction function to ensure it returns a float.
    """
    # 1. Create sample input data
    sample_input = {
        "OverallQual": 7, "GrLivArea": 1710, "GarageCars": 2,
        "TotalBsmtSF": 856, "FullBath": 2, "YearBuilt": 2003,
        # Add other necessary fields for engineer_features
        "1stFlrSF": 856, "2ndFlrSF": 854, "YrSold": 2008, "YearRemodAdd": 2003,
        "BsmtFinSF1": 706, "BsmtFinSF2": 0, "BsmtUnfSF": 150
    }
    
    # 2. Instantiate the mock objects
    model = MockModel()
    preprocessor = MockPreprocessor()

    # 3. Run the prediction function
    prediction = make_prediction(sample_input, model, preprocessor)

    # 4. Assert that the output is a float
    assert isinstance(prediction, float)
    # 5. Assert that the prediction is not None
    assert prediction is not None