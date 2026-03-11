import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Uber Fare Prediction", page_icon="🚕", layout="wide")
st.title("🚕 Uber Fare Prediction")

uploaded_file = st.file_uploader("Upload uber.csv", type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("1. Data Exploration")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.dataframe(df.describe())

    st.header("2. Data Cleaning")
    before = df.shape[0]
    df.dropna(inplace=True)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df.dropna(subset=['pickup_datetime'], inplace=True)
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 500)]
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['pickup_latitude'].between(-90, 90)) &
            (df['pickup_longitude'].between(-180, 180)) &
            (df['dropoff_latitude'].between(-90, 90)) &
            (df['dropoff_longitude'].between(-180, 180))]
    after = df.shape[0]
    st.write(f"Rows before: {before} | After: {after} | Removed: {before - after}")

    st.header("3. Feature Engineering")

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return 6371 * c

    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                  df['dropoff_latitude'], df['dropoff_longitude'])
    df = df[(df['distance_km'] > 0) & (df['distance_km'] <= 200)]

    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

    st.dataframe(df.head())

    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
        st.write("Sampled 50000 rows for faster training")

    st.header("4. EDA")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.hist(df['fare_amount'], bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('Fare Distribution')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sample = df.sample(min(5000, len(df)))
        ax2.scatter(sample['distance_km'], sample['fare_amount'], alpha=0.3, s=10)
        ax2.set_title('Fare vs Distance')
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        df.groupby('hour')['fare_amount'].mean().plot(kind='bar', color='coral', ax=ax3)
        ax3.set_title('Avg Fare by Hour')
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        df.groupby('day_of_week')['fare_amount'].mean().plot(kind='bar', color='lightgreen', ax=ax4)
        ax4.set_title('Avg Fare by Day')
        st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['fare_amount', 'passenger_count', 'distance_km', 'hour', 'day_of_week', 'month']].corr(),
                annot=True, cmap='coolwarm', fmt='.2f', ax=ax5)
    st.pyplot(fig5)

    st.header("5. Model Training")

    features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
                'passenger_count', 'distance_km', 'hour', 'day_of_week', 'month']

    X = df[features]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

    with st.spinner("Training Linear Regression..."):
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
    st.write("Linear Regression trained")

    with st.spinner("Training Random Forest..."):
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
    st.write("Random Forest trained")

    with st.spinner("Training XGBoost..."):
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
    st.write("XGBoost trained")

    st.success("All models trained!")

    st.header("6. Model Comparison")

    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
        'RMSE': [np.sqrt(mean_squared_error(y_test, lr_pred)),
                 np.sqrt(mean_squared_error(y_test, rf_pred)),
                 np.sqrt(mean_squared_error(y_test, xgb_pred))],
        'MAE': [mean_absolute_error(y_test, lr_pred),
                mean_absolute_error(y_test, rf_pred),
                mean_absolute_error(y_test, xgb_pred)],
        'R2': [r2_score(y_test, lr_pred),
               r2_score(y_test, rf_pred),
               r2_score(y_test, xgb_pred)]
    })
    st.dataframe(results)

    best = results.loc[results['RMSE'].idxmin(), 'Model']
    st.success(f"Best Model: {best}")

    st.header("7. Actual vs Predicted")
    col8, col9, col10 = st.columns(3)
    with col8:
        fig9, ax9 = plt.subplots()
        ax9.scatter(y_test, lr_pred, alpha=0.3, s=10)
        ax9.plot([0, 80], [0, 80], 'r--')
        ax9.set_title('Linear Regression')
        st.pyplot(fig9)
    with col9:
        fig10, ax10 = plt.subplots()
        ax10.scatter(y_test, rf_pred, alpha=0.3, s=10, color='green')
        ax10.plot([0, 80], [0, 80], 'r--')
        ax10.set_title('Random Forest')
        st.pyplot(fig10)
    with col10:
        fig11, ax11 = plt.subplots()
        ax11.scatter(y_test, xgb_pred, alpha=0.3, s=10, color='crimson')
        ax11.plot([0, 80], [0, 80], 'r--')
        ax11.set_title('XGBoost')
        st.pyplot(fig11)

    st.header("8. Feature Importance")
    col11, col12 = st.columns(2)
    with col11:
        fig12, ax12 = plt.subplots()
        pd.Series(rf.feature_importances_, index=features).sort_values().plot(kind='barh', color='green', ax=ax12)
        ax12.set_title('Random Forest')
        st.pyplot(fig12)
    with col12:
        fig13, ax13 = plt.subplots()
        pd.Series(xgb.feature_importances_, index=features).sort_values().plot(kind='barh', color='crimson', ax=ax13)
        ax13.set_title('XGBoost')
        st.pyplot(fig13)

    st.header("9. Predict Fare")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        pickup_lat = st.number_input("Pickup Latitude", value=40.7128)
        pickup_lon = st.number_input("Pickup Longitude", value=-74.0060)
        dropoff_lat = st.number_input("Dropoff Latitude", value=40.7589)
        dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9851)
    with pcol2:
        passengers = st.slider("Passengers", 1, 6, 1)
        trip_hour = st.slider("Hour", 0, 23, 12)
        trip_day = st.selectbox("Day", [0, 1, 2, 3, 4, 5, 6],
                                format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        trip_month = st.slider("Month", 1, 12, 6)

    if st.button("Predict"):
        try:
            dist = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

            input_data = pd.DataFrame([[pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                                        passengers, dist, trip_hour, trip_day, trip_month]],
                                      columns=features)

            lr_fare = lr.predict(scaler.transform(input_data))[0]
            rf_fare = rf.predict(input_data)[0]
            xgb_fare = xgb.predict(input_data)[0]

            st.success("Prediction done")
            st.write("Distance (km):", round(dist, 2))
            st.write("Linear Regression:", round(lr_fare, 2))
            st.write("Random Forest:", round(rf_fare, 2))
            st.write("XGBoost:", round(xgb_fare, 2))

        except Exception as e:
            st.error("Prediction failed")
            st.write(e)

else:
    st.info("Upload your uber.csv file to get started")
