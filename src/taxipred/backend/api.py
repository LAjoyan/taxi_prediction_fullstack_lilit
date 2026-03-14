from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.taxipred.utils.constants import DATA_PATH, USD_TO_SEK
from src.taxipred.backend.data_processing import build_features
from src.taxipred.backend.ors_routes import geocode_address, get_route
import requests


app = FastAPI(title="Taxi Price Prediction API", version="1.0")
router = APIRouter(prefix="/api/taxi/v1", tags=["taxi"])

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.joblib"
TRAIN_PATH = DATA_PATH / "df_train.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training data not found at: {TRAIN_PATH}")


model = joblib.load(MODEL_PATH)
df_train = pd.read_csv(TRAIN_PATH)


class TripInput(BaseModel):
    Trip_Distance_km: float = Field(gt=0, description="Distance in kilometers")
    Trip_Duration_Minutes: float = Field(gt=0, description="Duration in minutes")
    Time_of_Day: str = Field(description="Afternoon, Evening, Morning, Night, Unknown")
    Day_of_Week: str = Field(description="Weekday, Weekend, Unknown")
    Traffic_Conditions: str = Field(description="High, Medium, Low, Unknown")
    Weather: str = Field(description="Clear, Rain, Snow, Unknown")


class PredictionOutput(BaseModel):
    estimated_price: float
    currency: str
    predicted_price_log: float


class RouteRequest(BaseModel):
    from_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)


class RouteResponse(BaseModel):
    distance_km: float
    duration_min: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    polyline_latlon: list[list[float]]


@router.get("/")
def root():
    return {"message": "Taxi Price Prediction API is running"}


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "train_rows": int(df_train.shape[0]),
    }


@router.get("/data/sample")
def get_data_sample(rows: int = 5):
    return df_train.head(rows).to_dict(orient="records")


@router.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    try:
        start = geocode_address(req.from_address)
        end = geocode_address(req.to_address)
        result = get_route(start, end)
        return result

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="OpenRouteService timed out. Try again or increase timeout.",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ORS request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict", response_model=PredictionOutput)
def predict(payload: TripInput):
    try:
        X_in = build_features(payload.model_dump())

        pred_log = float(model.predict(X_in)[0])
        pred_price_usd = float(np.expm1(pred_log))
        pred_price_sek = pred_price_usd * USD_TO_SEK

        return {
            "estimated_price": round(pred_price_sek, 2),
            "currency": "SEK",
            "predicted_price_log": pred_log,
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal prediction error: {str(e)}"
        )


app.include_router(router)
