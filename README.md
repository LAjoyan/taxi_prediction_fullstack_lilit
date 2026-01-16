# Taxi Trip Prediction System (Lilit)
Fullstack ML application for predicting taxi prices.

## 📁 Project Structure
* **src/taxipred/backend**: FastAPI web server logic.
* **src/taxipred/frontend**: Streamlit interface for user interaction.
* **src/taxipred/model_development**: Jupyter notebooks for EDA and model training.
* **src/taxipred/data**: Local storage for raw and processed datasets (`df_train.csv`, `df_predict.csv`).

## 📊 Data Processing Pipeline (Uppgift 1)

### 1. Data Cleaning & Feature Engineering
* **Initial Cleanup**: Dropped the `Passenger_Count` column as it does not significantly affect the target variable `Trip_Price`.
* **Mathematical Imputation**: Recovered missing values for `Trip_Distance_km`, `Base_Fare`, `Per_Km_Rate`, and `Per_Minute_Rate` by solving the pricing equation: 
  $$Trip\_Price = Base\_Fare + (Distance \times Km\_Rate) + (Duration \times Minute\_Rate)$$
* **Categorical Handling**: Filled missing values in `Time_of_Day`, `Day_of_Week`, `Traffic_Conditions`, and `Weather` with the label **"Unknown"** to preserve data rows for the model.

### 2. Outlier Removal (IQR Method)
* **Statistical Filtering**: Used the Interquartile Range (IQR) to identify and handle extreme anomalies in the training set.
* **Constraints**: Applied strict filtering to include only realistic urban trips:
  * `Trip_Distance_km <= 50`
  * `Trip_Price <= 150`
* **Result**: Reduced the training set to **916 high-quality entries**, preventing the model from being skewed by rare, extreme long-distance trips.

### 3. Feature Transformation
* **Target Normalization**: Applied a log-transform ($np.log1p$) to create `Trip_Price_log`. This reduces skewness and helps the Linear Regression model achieve higher accuracy.
* **One-Hot Encoding**: Converted categorical text into numeric format using `pd.get_dummies(drop_first=True, dtype=float)`.
* **Dtype Consistency**: Ensured all 14 feature columns are converted to `float64` for direct compatibility with Scikit-Learn.

### 4. Data Alignment & Export
* **Synchronization**: Implemented a final alignment step to ensure the prediction set (`df_predict.csv`) contains the **exact same columns and order** as the training features.
* **Final Outputs**:
  * `df_train.csv`: 916 rows (Features + `Trip_Price` + `Trip_Price_log`).
  * `df_predict.csv`: 32 rows (Features only, aligned for prediction).