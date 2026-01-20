```text
taxi_prediction_fullstack_lilit/
├── .venv/
├── src/
│   └── taxipred/
│       ├── backend/
│       │   ├── api.py
│       │   ├── data_processing.py
│       │   └── random_forest_model.joblib
│       ├── data/
│       │   ├── screenshots/
│       │   │   ├── correlation_heatmap_encoded.png
│       │   │   ├── eda_cleaned_price_distance.png
│       │   │   ├── eda_outliers.png
│       │   │   ├── streamlit_prediction.png
│       │   │   └── streamlit_ui.png
│       │   ├── df_predict.csv
│       │   ├── df_train.csv
│       │   └── taxi_trip_pricing.csv
│       ├── frontend/
│       │   └── app.py
│       ├── model_development/
│       │   ├── eda.ipynb
│       │   └── model_dev.ipynb
│       └── utils/
│           ├── __init__.py
│           └── constants.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock


```

# 🚕 Taxi Trip Prediction System

A full-stack machine learning application for predicting taxi trip prices based on trip
distance, duration, traffic conditions, time of day, day of week, and weather.

---

## 📌 Project Overview

This project focuses on building a robust price prediction system using real-world–style
taxi trip data.  
The pipeline includes data cleaning, mathematical imputation, feature engineering,
outlier handling, model training, and deployment through a decoupled backend and frontend.

---

## 🗂 Project Structure

- **src/taxipred/backend** – FastAPI backend serving the trained model
- **src/taxipred/frontend** – Streamlit application for user interaction
- **src/taxipred/model_development** – Jupyter notebooks for EDA and model training
- **src/taxipred/data** – Raw and processed datasets + screenshots (EDA + Streamlit UI)

---

## 📊 Data Processing Pipeline

### 1. Data Cleaning

- Removed `Passenger_Count` due to negligible impact on price
- Identified missing values across numerical and categorical features
- Dropped rows where multiple critical values were missing
- Filled missing categorical values with `"Unknown"` to preserve data

---

### 2. Mathematical Imputation

Missing numerical values were recovered using the pricing equation:


### Option 3 (if you want bullets): don’t bold the bullets
Paste this:

```md
Trip_Price = Base_Fare  
- (Trip_Distance_km × Per_Km_Rate)  
- (Trip_Duration_Minutes × Per_Minute_Rate)
```


This approach minimized data loss while maintaining numerical accuracy.

---

### 3. Feature Selection

After imputation, the following columns were removed:

- `Base_Fare`
- `Per_Km_Rate`
- `Per_Minute_Rate`

**Reasoning:**

- Prevents data leakage
- Avoids perfect multicollinearity
- Forces the model to learn real-world pricing patterns

**Final features used:**

- Trip_Distance_km
- Trip_Duration_Minutes
- Time_of_Day
- Day_of_Week
- Traffic_Conditions
- Weather

---

### 4. Outlier Handling

- Applied Interquartile Range (IQR) filtering
- Removed unrealistic high-distance and high-price trips
- Final training dataset size: **916 rows**

---

### 5. Feature Encoding & Alignment

- One-hot encoded categorical variables
- Applied log transformation (`log1p`) to the target variable
- Ensured training and prediction datasets had identical feature structure

---

### 6. Statistical Validation & Integrity Check

Before finalizing the data for training, a rigorous statistical audit was performed using descriptive statistics (.describe().T):

- Outlier Mitigation: Verified that all entries in the final training set strictly adhere to defined thresholds.

- Normalization: Confirmed that the log-transformation successfully addressed skewness in the target variable.

- Feature Consistency: Verified that one-hot encoded variables are correctly scaled between 0.0 and 1.0.

- Schema Alignment: Ensured that df_predict.csv contains the exact same 14-feature statistical baseline as the training set.

---

#### Outlier Identification & Removal
This comparison shows how we eliminated noise by capping distances at 50km and prices at $150.

Below are key EDA plots used to justify cleaning and feature engineering.

![Outliers Before](src/taxipred/data/screenshots/eda_outliers.png)
![Cleaned Data](src/taxipred/data/screenshots/eda_cleaned_price_distance.png)

#### Feature Relationships
The heatmap below validates our feature selection and highlights the strong correlation between distance and price.
![Correlation Heatmap](src/taxipred/data/screenshots/correlation_heatmap_encoded.png)
---


## 🤖 Model Development & Evaluation

Model development followed a structured workflow: **baseline → linear model → tree-based model**.
Models were compared using **MAE, RMSE, and R²** on a held-out test set (and/or cross-validation).
The final model was trained on a **log-transformed target (`log1p(Trip_Price)`)** to reduce skewness
and improve numerical stability.

### Baseline Model

A simple baseline model was established using the median of the target variable.
This model serves as a reference point to ensure that any trained machine learning
model provides meaningful improvement over naive predictions.

### Linear Regression

Linear Regression was used as the first machine learning model due to its
simplicity and interpretability. The model was trained on the cleaned and
feature-engineered dataset and evaluated on a held-out test set.

Model performance was assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

The Linear Regression model significantly outperformed the baseline, achieving
a positive R² score and substantially lower error metrics, indicating that the
engineered features capture meaningful pricing patterns.

### Feature Scaling

Feature scaling was evaluated but not applied in the final pipeline.
The selected models (Linear Regression as a reference model and Random Forest
as a non-linear model) do not require feature scaling to perform correctly.

### Random Forest Regressor

Random Forest was evaluated to capture non-linear pricing effects and feature
interactions that linear models cannot represent. As a tree-based model, Random
Forest does not require feature scaling.

Compared to Linear Regression, Random Forest achieved lower error metrics and a
higher R² score, indicating improved predictive performance.

### Final Model Selection

Linear Regression was used as a reference model due to its simplicity and
interpretability. Random Forest achieved the best overall performance with
lower error metrics and a higher R² score.

Therefore, Random Forest was selected as the final model.

The selected Random Forest model was retrained on the full dataset before export.

### Model Export

After final model selection, the Random Forest model was retrained on the full
dataset to maximize learning from all available data.

The trained model was then exported using `joblib` to enable reuse during
deployment and inference:

- **Model file:** `random_forest_model.joblib`
- **Location:** `src/taxipred/backend/`

This file is later loaded by the backend service to serve price predictions
without retraining the model.

---

## 🌐 Application

## 🧰 Tech stack
- Python, FastAPI (backend API)
- Streamlit (frontend UI)
- scikit-learn + joblib (model + export)
- pandas + numpy (data processing)


## ▶ Run the full app (backend + frontend)

1) Start backend (FastAPI)
2) Start frontend (Streamlit)

The Streamlit UI will call the FastAPI API for predictions.

## ▶ Run the Backend (FastAPI)

From the project root:

```bash
uv run uvicorn taxipred.backend.api:app --reload
```
Open Swagger UI: 
http://127.0.0.1:8000/docs 

### API Endpoints
- `GET /api/taxi/v1/health` – backend status
- `GET /api/taxi/v1/data/sample` – sample rows from training data
- `POST /api/taxi/v1/predict` – price prediction


## ▶ Run the Frontend (Streamlit)

Start the backend first (FastAPI), then in a new terminal run:

```bash
uv run streamlit run src/taxipred/frontend/app.py
```
Open the app in your browser:

- Streamlit UI: http://localhost:8501

**Note:** The Streamlit app calls the FastAPI endpoint:
`POST /api/taxi/v1/predict`

---

## 🧱 Architecture (High-level)

**Streamlit (Frontend)** collects user trip inputs and sends them to the backend.

**FastAPI (Backend)**:
- validates the input (Pydantic)
- transforms input into the exact model feature format (`data_processing.py`)
- loads the trained Random Forest model (`random_forest_model.joblib`)
- returns the predicted trip price

Flow:
**Streamlit → FastAPI `/api/taxi/v1/predict` → RandomForest model → Predicted price**

---

## 🖥️ Frontend (Streamlit)

### Streamlit UI
![Streamlit UI](src/taxipred/data/screenshots/streamlit_ui.png)

### Example prediction result

![Streamlit Prediction](src/taxipred/data/screenshots/streamlit_prediction.png)

---

## 📦 Outputs
- `df_train.csv` – Cleaned and processed training dataset
- `df_predict.csv` – Aligned prediction dataset (32 trips)

---

## ⚠ Limitations
- Original pricing rate features were removed to prevent leakage
- Linear regression may not capture complex non-linear pricing behavior

---

## 🧠 Reflections / What I learned

- How to structure a full ML project with a clean separation between model development, backend API, and frontend UI.
- How to prevent data leakage (feature selection + consistent feature schema).
- How to serve a trained ML model with FastAPI and consume it from Streamlit.
- The importance of validating user input and transforming it into the exact feature format the model expects.
