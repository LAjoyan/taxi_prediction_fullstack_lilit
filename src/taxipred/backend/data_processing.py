from taxipred.utils.constants import DATA_PATH
import pandas as pd

# Load training schema once to get the exact feature columns the model expects
_df_train = pd.read_csv(DATA_PATH / "df_train.csv")

if "Trip_Price" not in _df_train.columns:
    raise ValueError(
        "df_train.csv must contain 'Trip_Price' column to infer FEATURE_COLUMNS."
    )

FEATURE_COLUMNS = [
    c for c in _df_train.columns if c not in ["Trip_Price", "Trip_Price_log"]
]


def build_features(payload: dict) -> pd.DataFrame:
    features = {col: 0.0 for col in FEATURE_COLUMNS}

    # Required numeric fields
    features["Trip_Distance_km"] = float(payload["Trip_Distance_km"])
    features["Trip_Duration_Minutes"] = float(payload["Trip_Duration_Minutes"])

    # Categorical fields
    time_of_day = payload.get("Time_of_Day", "Unknown")
    day_of_week = payload.get("Day_of_Week", "Unknown")
    traffic = payload.get("Traffic_Conditions", "Unknown")
    weather = payload.get("Weather", "Unknown")

    # Set one-hot columns only if they exist in FEATURE_COLUMNS
    col = f"Time_of_Day_{time_of_day}"
    if col in features:
        features[col] = 1.0

    col = f"Day_of_Week_{day_of_week}"
    if col in features:
        features[col] = 1.0

    col = f"Traffic_Conditions_{traffic}"
    if col in features:
        features[col] = 1.0

    col = f"Weather_{weather}"
    if col in features:
        features[col] = 1.0

    return pd.DataFrame([features])[FEATURE_COLUMNS]
