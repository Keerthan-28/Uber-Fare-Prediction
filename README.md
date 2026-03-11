# 🚕 Uber Fare Prediction

A machine learning project that predicts Uber taxi fare prices based on trip details like pickup/dropoff coordinates, passenger count and distance.

## 📌 Problem Statement

Predict taxi fare price based on trip details using regression models.

## 📂 Dataset

The dataset contains Uber ride information with the following columns:

| Column | Description |
|--------|-------------|
| fare_amount | Fare price in dollars |
| pickup_datetime | Date and time of pickup |
| pickup_longitude | Longitude of pickup location |
| pickup_latitude | Latitude of pickup location |
| dropoff_longitude | Longitude of dropoff location |
| dropoff_latitude | Latitude of dropoff location |
| passenger_count | Number of passengers |

## 🔧 Steps Followed

### 1. Data Loading
Loaded the uber.csv dataset using pandas.

### 2. Data Exploration
Checked shape, data types, statistical summary and missing values to understand the dataset.

### 3. Data Cleaning
- Removed rows with missing values
- Converted pickup_datetime from text to proper datetime format
- Removed outliers like negative fares, 0 passengers, invalid coordinates

### 4. Feature Engineering
- Created **distance_km** feature using **Haversine formula** which calculates the great circle distance between pickup and dropoff points on Earth's surface
- Extracted **hour**, **day**, **month**, **year**, **day_of_week** from pickup_datetime

### 5. Exploratory Data Analysis
- Fare distribution histogram
- Fare vs Distance scatter plot
- Average fare by hour of day
- Average fare by day of week
- Correlation heatmap

### 6. Model Training
Trained 3 regression models:

| Model | How It Works |
|-------|-------------|
| **Linear Regression** | Fits a straight line through the data. Used as a baseline model |
| **Random Forest** | Creates 100 independent decision trees and averages their predictions |
| **XGBoost** | Builds 200 sequential trees where each tree corrects the previous tree's mistakes |

### 7. Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **RMSE** | Average error where large mistakes are penalized more. Lower is better |
| **MAE** | Average error where all mistakes are treated equally. Lower is better |
| **R² Score** | How much of the fare variation the model explains. Higher is better (max 1.0) |

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit

## 📁 Project Structure

```
├── app.py               # Streamlit web app
├── uber.csv             # Dataset
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── notebooks/
    └── uber_fare.ipynb  # Google Colab notebook
```

## 🚀 How to Run Locally

```bash
git clone https://github.com/Keerthan-28/uber-fare-prediction.git
cd uber-fare-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

Deployed on Streamlit Cloud. Upload uber.csv on the app and it runs the full pipeline automatically.

[Click here to open the app](https://your-app-link.streamlit.app)

## 📊 Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | ~ | ~ | ~ |
| Random Forest | ~ | ~ | ~ |
| XGBoost | ~ | ~ | ~ |

> XGBoost performed best with lowest RMSE and highest R² score

## 🔑 Key Findings

1. **Distance** is the most important feature for predicting fare
2. Tree based models (Random Forest, XGBoost) outperform Linear Regression because the relationship between fare and distance is not perfectly linear
3. Time features like hour and day add value because of surge pricing and traffic patterns
4. Passenger count has very low impact on fare price

## 📈 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Add more features like weather and traffic data
- Try other models like LightGBM and CatBoost
- Use cross validation instead of single train test split

## 👤 Author

**Keerthan**
- GitHub: [@Keerthan-28](https://github.com/Keerthan-28)
