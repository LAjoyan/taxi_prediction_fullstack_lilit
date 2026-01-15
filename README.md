# taxi_prediction_fullstack_lilit
Fullstack ML application for predicting taxi prices 


# Taxi Trip Prediction System

This project is a full-stack application designed to predict taxi trip outcomes. It includes a data processing pipeline, a machine learning model, a backend API, and a frontend interface.

## 📁 Project Structure
* **src/taxipred/backend**: FastAPI/Flask web server logic.
* **src/taxipred/frontend**: Streamlit or web interface.
* **src/taxipred/model_development**: Jupyter notebooks for EDA and model training.
* **src/taxipred/utils**: Shared constants and helper functions.
* **src/taxipred/data**: Local storage for raw and processed datasets.

## 🛠 Setup & Installation

This project uses `uv` for Python package management.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd TAXI_PREDICTION_FULLSTACK_LILIT
    ```


## Data Loading
- Loaded the taxi trip dataset (`taxi_trip_pricing.csv`) into a DataFrame.

## Initial Exploration
- Checked dataset info and missing values.
- Dropped `Passenger_Count` column as it does not significantly affect `Trip_Price`.

## Handling Missing Values
- Dropped rows with missing critical values for ML (`Trip_Distance_km`, `Base_Fare`, `Per_Km_Rate`, `Per_Minute_Rate`, `Trip_Duration_Minutes`) where they could not be reliably calculated.
- Calculated missing values where possible:
  - `Trip_Distance_km` from `Trip_Price`, `Base_Fare`, `Per_Km_Rate`, `Per_Minute_Rate`, `Trip_Duration_Minutes`
  - `Base_Fare` from `Trip_Price`, `Trip_Distance_km`, `Per_Km_Rate`, `Trip_Duration_Minutes`, `Per_Minute_Rate`
  - `Per_Km_Rate` from `Trip_Price`, `Base_Fare`, `Trip_Distance_km`, `Per_Minute_Rate`, `Trip_Duration_Minutes`
  - `Per_Minute_Rate` from `Trip_Price`, `Base_Fare`, `Trip_Distance_km`, `Per_Km_Rate`, `Trip_Duration_Minutes`
- Removed rows that still had missing values after calculation.

## Resulting Dataset
- Final cleaned dataset has 974 entries and no missing values for essential features.
- Ready for further EDA and visualization.

## Target Transformation

- The `Trip_Price` distribution is heavily right-skewed due to a few very high-priced trips.
- Applied a log-transform (`log1p`) to create a new column `Trip_Price_log`.
- This reduces skew and makes the data more suitable for regression models.

