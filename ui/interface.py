# ui/interface.py

import streamlit as st
import requests
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --- HEADER ---
st.title("AI-Powered Real Estate Price Predictor 🏡")
st.markdown("""
Welcome to the Ames Housing Price Predictor! This application leverages a sophisticated **XGBoost machine learning model** to provide an estimated sale price for residential properties in Ames, Iowa.

**How to use this tool:**
1.  Use the **sidebar on the left** to adjust the key features of a house.
2.  The most impactful features (like overall quality and living area) are available for you to modify.
3.  Click the **"Predict Price"** button to see the model's prediction.

""")

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("House Features")

# --- DEFAULT DATA (for reference and non-user-facing fields) ---
default_data = {
    "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450,
    "Street": "Pave", "Alley": None, "LotShape": "Reg", "LandContour": "Lvl",
    "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl",
    "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm",
    "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": 7,
    "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003,
    "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0,
    "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd",
    "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 706, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 150,
    "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr",
    "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0,
    "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2,
    "HalfBath": 1, "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd",
    "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": None,
    "GarageType": "Attchd", "GarageYrBlt": 2003.0, "GarageFinish": "RFn",
    "GarageCars": 2, "GarageArea": 548, "GarageQual": "TA", "GarageCond": "TA",
    "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0, # <-- TYPO FIX
    "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": None,
    "Fence": None, "MiscFeature": None, "MiscVal": 0, "MoSold": 2, "YrSold": 2008,
    "SaleType": "WD", "SaleCondition": "Normal"
}

# --- USER INPUTS ---
st.sidebar.subheader("Key Features")
# Create input fields and store their values directly
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, default_data["OverallQual"])
gr_liv_area = st.sidebar.number_input("Above Grade Living Area (sq ft)", min_value=0, value=default_data["GrLivArea"])
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 5, default_data["GarageCars"])
garage_area = st.sidebar.number_input("Garage Area (sq ft)", min_value=0, value=default_data["GarageArea"])
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, value=(default_data["BsmtFinSF1"] + default_data["BsmtFinSF2"] + default_data["BsmtUnfSF"]))
first_flr_sf = st.sidebar.number_input("First Floor Area (sq ft)", min_value=0, value=default_data["1stFlrSF"])
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, default_data["FullBath"])
tot_rms_abv_grd = st.sidebar.slider("Total Rooms Above Grade", 1, 15, default_data["TotRmsAbvGrd"])
year_built = st.sidebar.slider("Year Built", 1870, 2025, default_data["YearBuilt"])
year_remod_add = st.sidebar.slider("Year Remodeled", 1950, 2025, default_data["YearRemodAdd"])

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price"):
    # 1. Create the payload for the API
    # Start with the default data and update it with user inputs
    api_payload = default_data.copy()
    
    # Update fields with user-provided values
    api_payload["OverallQual"] = overall_qual
    api_payload["GrLivArea"] = gr_liv_area
    api_payload["GarageCars"] = garage_cars
    api_payload["GarageArea"] = garage_area
    api_payload["1stFlrSF"] = first_flr_sf
    api_payload["FullBath"] = full_bath
    api_payload["TotRmsAbvGrd"] = tot_rms_abv_grd
    api_payload["YearBuilt"] = year_built
    api_payload["YearRemodAdd"] = year_remod_add
    
    # Handle derived/dependent fields
    api_payload["BsmtUnfSF"] = total_bsmt_sf
    api_payload["BsmtFinSF1"] = 0
    api_payload["BsmtFinSF2"] = 0
    # A simple assumption for 2nd floor SF
    second_flr_sf = gr_liv_area - first_flr_sf
    api_payload["2ndFlrSF"] = second_flr_sf if second_flr_sf > 0 else 0

    # 2. Send the request to the FastAPI backend
    try:
        api_url = "http://predictor-prod-env.eba-hetgzns3.us-east-1.elasticbeanstalk.com/predict"
        response = requests.post(api_url, data=json.dumps(api_payload))
        response.raise_for_status()

        # 3. Display the result
        result = response.json()
        price = result.get("predicted_price_formatted", "N/A")
        
        st.subheader("Predicted Sale Price:")
        st.success(f"**{price}**")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Please ensure it is running. Error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
