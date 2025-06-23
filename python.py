# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

# Title
st.title("📈 Tesla Stock Price Prediction using Machine Learning")

# File uploader
uploaded_file = st.file_uploader("Upload Tesla CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # Check for necessary columns
    if 'Date' in df.columns and 'Close' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        st.subheader("📉 Closing Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label="Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Prepare features and labels
        X = df[['Days']]
        y = df['Close']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("📈 Model Performance")
        st.write(f"Mean Squared Error on Test Data: {mse:.2f}")

        # Plot predictions vs actual
        st.subheader("🔍 Predictions vs Actual")
        fig2, ax2 = plt.subplots()
        ax2.scatter(X_test, y_test, label="Actual", color="blue", alpha=0.6)
        ax2.plot(X_test, y_pred, label="Predicted", color="red")
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)

        # User input for prediction
        st.subheader("📅 Predict Future Stock Price")
        future_days = st.slider("Select number of days from start date", 1, 5000, 1500)
        future_price = model.predict(np.array([[future_days]]))[0]
        st.write(f"Predicted Closing Price after {future_days} days: *${future_price:.2f}*")

    else:
        st.error("CSV must contain 'Date' and 'Close' columns.")

else:
    st.info("Please upload a Tesla stock CSV file to begin.")
