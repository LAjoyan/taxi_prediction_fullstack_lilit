from taxipred.utils.constants import DATA_PATH
import pandas as pd

_df_train = pd.read_csv(DATA_PATH / 'df_train.csv')

FEATURE_COLUMNS = [
    c for c in _df_train.columns if c not in ['Trip_Price', 'Trip_Price_log']
]

required_numeric = {'Trip_Distance_km', 'Trip_Duration_Minutes'}
missing_required = required_numeric - set(FEATURE_COLUMNS)
if missing_required:
    raise ValueError(f'df_train.csv is missing required feature columns: {sorted(missing_required)}')

if len(FEATURE_COLUMNS) == 0:
    raise ValueError('No feature columns found. Check df_train.csv export and FEATURE_COLUMNS logic.')

def _clean_cat(value: str) -> str:
    if value is None:
        return 'Unknown'
    return str(value).strip() or 'Unknown'

def build_features(payload: dict) -> pd.DataFrame:
    features = {col: 0.0 for col in FEATURE_COLUMNS}

    features['Trip_Distance_km'] = float(payload['Trip_Distance_km'])
    features['Trip_Duration_Minutes'] = float(payload['Trip_Duration_Minutes'])

    time_of_day = _clean_cat(payload.get('Time_of_Day'))
    day_of_week = _clean_cat(payload.get('Day_of_Week'))
    traffic = _clean_cat(payload.get('Traffic_Conditions'))
    weather = _clean_cat(payload.get('Weather'))

    for colname in (
        f'Time_of_Day_{time_of_day}',
        f'Day_of_Week_{day_of_week}',
        f'Traffic_Conditions_{traffic}',
        f'Weather_{weather}',
    ):
        if colname in features:
            features[colname] = 1.0

    return pd.DataFrame([features])[FEATURE_COLUMNS]