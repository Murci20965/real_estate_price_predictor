# tests/test_preprocessing.py

import pandas as pd
from src.preprocessing import engineer_features

def test_engineer_features():
    """
    Tests that the engineer_features function correctly adds the expected new columns.
    """
    # 1. Create a sample DataFrame with the necessary input columns
    sample_data = {
        'TotalBsmtSF': [1000],
        '1stFlrSF': [1200],
        '2ndFlrSF': [800],
        'YrSold': [2010],
        'YearBuilt': [2000],
        'YearRemodAdd': [2005]
    }
    df = pd.DataFrame(sample_data)

    # 2. Run the function
    df_engineered = engineer_features(df)

    # 3. Assert that the new columns exist
    assert 'TotalSF' in df_engineered.columns
    assert 'HouseAge' in df_engineered.columns
    assert 'WasRemodeled' in df_engineered.columns

    # 4. Assert that the calculations are correct
    assert df_engineered['TotalSF'].iloc[0] == 3000
    assert df_engineered['HouseAge'].iloc[0] == 10
    assert df_engineered['WasRemodeled'].iloc[0] == 1