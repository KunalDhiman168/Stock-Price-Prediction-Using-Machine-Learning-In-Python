# Stock-Price-Prediction-Using-Machine-Learning-In-Python
 This project demonstrates the use of machine learning algorithms to predict stock price movements and closing prices based on historical market data. It aims to provide insights into future stock trends using regression and classification techniques, making it a helpful tool for financial forecasting and data-driven investment decisions.

# Project Overview
Predicting stock prices is a complex task influenced by multiple factors like historical performance, volume, and technical indicators. In this project, we use K-Nearest Neighbors (KNN) to:

- Classify whether the stock's price will go up or down.

- Predict the exact closing price for a given day.

Historical data of Tata Consumer Products Ltd (TATACONSUM.NS) has been sourced from Yahoo Finance, which includes attributes like open, close, high, low, volume, etc.

 # Key Features
 - Fetches historical stock data using yfinance.

 - Performs extensive data preprocessing and feature engineering.

 - Implements K-Nearest Neighbors (KNN) for both classification and regression.

 - Evaluates model performance using accuracy scores and error metrics like MAE & RMSE.

 - Visualizes real vs predicted stock prices for better understanding.

 - Offers insights into feature impact on model performance.

# Technologies Used
- **Python** <br>

- **Pandas** – for data manipulation <br>

- **NumPy** – for numerical computations <br>

- **Scikit-learn** – for machine learning models <br>

- **Matplotlib**  – for data visualization <br>

- **yfinance** – for fetching historical stock data <br>

# Model Approach
**1. Data Collection**
Used yfinance to download stock data of Tata Consumer Products (TATACONSUM.NS).

Data includes: Open, High, Low, Close, Volume, and Adjusted Close.

**2. Feature Engineering**
Calculated daily returns.

Created lag features.

Constructed target variables for classification (Up/Down) and regression (Closing Price).

**3. Data Splitting**
Split data into training and testing sets to validate model performance.

**4. Model Training**
Trained two models:

Classification: Predict if the next day’s price will be higher or lower.

Regression: Predict the actual closing price.

**5. Model Evaluation**
Used:

Accuracy for classification.

MAE (Mean Absolute Error) and RMSE (Root Mean Square Error) for regression.

**6. Visualization**
Plotted graphs for actual vs predicted prices.

Displayed confusion matrix and accuracy metrics.

# Future Work
- Integrate LSTM (Long Short-Term Memory) neural networks for better time-series prediction.

- Add real-time stock prediction interface using a web app (e.g., Streamlit).

- Expand to multi-stock analysis and portfolio suggestions.








