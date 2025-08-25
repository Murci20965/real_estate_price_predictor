# app/main.py

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

# Import our custom modules
from src.logger_config import logger
from app.predict import load_latest_model, make_prediction

# --- APP SETUP ---
app = FastAPI(
    title="Real Estate Price Predictor API",
    description="API to predict house prices in Ames, Iowa.",
    version="0.1.0"
)

# --- GLOBAL VARIABLES ---
model = None
preprocessor = None

# --- API EVENTS ---
@app.on_event("startup")
def startup_event():
    """Load the model and preprocessor when the API starts."""
    global model, preprocessor
    logger.info("--- API starting up ---")
    model, preprocessor = load_latest_model()
    if model is None or preprocessor is None:
        logger.error("FATAL: Model or preprocessor could not be loaded. API will not work.")
    else:
        logger.info("Model and preprocessor loaded successfully.")

# --- DATA MODEL ---
class HouseData(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: str
    Alley: Optional[str] = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: int
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: int
    BsmtUnfSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str] = None
    first_flr_sf: int = Field(..., alias="1stFlrSF")
    second_flr_sf: int = Field(..., alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: int
    GarageArea: int
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    three_ssn_porch: int = Field(..., alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

    class Config:
        json_schema_extra = {
            "example": {
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
                "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0,
                "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": None,
                "Fence": None, "MiscFeature": None, "MiscVal": 0, "MoSold": 2, "YrSold": 2008,
                "SaleType": "WD", "SaleCondition": "Normal"
            }
        }

# --- API ENDPOINTS ---
@app.get("/", tags=["General"])
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "API is running!"}

@app.get("/health", tags=["Health Check"])
def health_check():
    """
    A more detailed health check that verifies the model is loaded.
    """
    if model is not None and preprocessor is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "model_loaded": False},
        )

@app.post("/predict", tags=["Prediction"])
def predict_price(house_data: HouseData):
    """Predicts the price of a house based on its features."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. API is not ready.")
    
    input_dict = house_data.dict(by_alias=True)
    
    prediction = make_prediction(input_dict, model, preprocessor)
    
    if prediction is None:
        raise HTTPException(status_code=500, detail="Prediction could not be made.")
    
    # Return the price in a formatted string as a response to the user in the frontend app (in dollars)
    return {
        "predicted_price_formatted": f"${prediction:,.2f}"
    }
