import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the optimized LSTM model
model = tf.keras.models.load_model("optimized_lstm_currency_model.h5")

# Streamlit UI for input and predictions
def predict_currency():
    # Title of the app
    st.title("Optimized Currency Exchange Rate Predictor with LSTM and HOA")

    # User input: Select currency pair
    currency_ticker = st.text_input("Enter Currency Pair (e.g., 'USDINR=X', 'GBPUSD=X')", "USDINR=X")
    
    # Date range for prediction
    start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2023-01-01'))

    # Download data based on user input
    currency_data = yf.download(currency_ticker, start=start_date, end=end_date, interval='1d')
    currency_data = currency_data[['Adj Close']].dropna()

    if len(currency_data) > 60:  # Ensure enough data for 60-day window
        # Preprocess data for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(currency_data[['Adj Close']].values)

        # Prepare sliding windows for predictions
        X_inputs = []
        for i in range(60, len(scaled_data)):
            X_inputs.append(scaled_data[i - 60:i])
        X_inputs = np.array(X_inputs).reshape(-1, 60, 1)

        # Predict prices for all valid rows
        predictions = model.predict(X_inputs)
        predicted_prices = scaler.inverse_transform(predictions).flatten()  # Reverse scaling to get actual prices

        # Prepare data for the table and chart
        actual_prices = currency_data['Adj Close'].values[60:]  # Actual prices from the 60th day onward
        dates = currency_data.index[60:].date  # Dates corresponding to predictions
        comparison_data = pd.DataFrame({
            "Date": dates,
            "Actual Prices": actual_prices,
            "Predicted Prices": predicted_prices
        })

        # Display the table
        st.subheader("Actual vs Predicted Prices")
        st.dataframe(comparison_data)

        # Plot the chart
        st.line_chart(comparison_data.set_index("Date")[["Actual Prices", "Predicted Prices"]])

    else:
        st.error("Not enough data to generate predictions (minimum 60 days of data required).")

# Call the prediction function to run the Streamlit app
predict_currency()