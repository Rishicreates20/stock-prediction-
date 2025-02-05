<<<<<<< HEAD
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="NVIDIA Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# Sidebar for User Input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="NVDA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=180, value=60)
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20) / 100

# Main App Title
st.title("NVIDIA Stock Price Prediction")
st.write("Predict future stock prices using LSTM neural networks.")

# Fetch Data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocess Data
def preprocess_data(data, lookback):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Ensure X has the correct shape for LSTM
    if len(X) == 0:
        st.error("Not enough data points for the specified lookback period. Try reducing the lookback period.")
        return None, None, None
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Evaluate Model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    return predicted_prices, actual_prices, mse, mae, r2, history

# Main App Logic
if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching data and training model..."):
        # Fetch data
        data = get_stock_data(ticker, start_date, end_date)
        
        # Preprocess data
        X, y, scaler = preprocess_data(data, lookback)
        
        if X is not None:  # Only proceed if preprocessing succeeded
            # Split data
            split = int((1 - test_size) * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build and train model
            model = build_lstm_model((X_train.shape[1], 1))
            predicted_prices, actual_prices, mse, mae, r2, history = train_and_evaluate(
                model, X_train, y_train, X_test, y_test, scaler
            )
            
            # Display results
            st.success("Prediction complete!")
            
            # Plot predictions
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(actual_prices, color='black', label="Actual Prices")
            ax.plot(predicted_prices, color='green', label="Predicted Prices")
            ax.set_title(f"{ticker} Stock Price Prediction")
            ax.set_xlabel("Time")
            ax.set_ylabel("Stock Price")
            ax.legend()
            st.pyplot(fig)
            
            # Plot training loss
            st.subheader("Training Progress")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title("Model Loss Progress")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Rishikesh")
=======
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="NVIDIA Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# Sidebar for User Input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="NVDA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=180, value=60)
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20) / 100

# Main App Title
st.title("NVIDIA Stock Price Prediction")
st.write("Predict future stock prices using LSTM neural networks.")

# Fetch Data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocess Data
def preprocess_data(data, lookback):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Ensure X has the correct shape for LSTM
    if len(X) == 0:
        st.error("Not enough data points for the specified lookback period. Try reducing the lookback period.")
        return None, None, None
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Evaluate Model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    return predicted_prices, actual_prices, mse, mae, r2, history

# Main App Logic
if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching data and training model..."):
        # Fetch data
        data = get_stock_data(ticker, start_date, end_date)
        
        # Preprocess data
        X, y, scaler = preprocess_data(data, lookback)
        
        if X is not None:  # Only proceed if preprocessing succeeded
            # Split data
            split = int((1 - test_size) * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build and train model
            model = build_lstm_model((X_train.shape[1], 1))
            predicted_prices, actual_prices, mse, mae, r2, history = train_and_evaluate(
                model, X_train, y_train, X_test, y_test, scaler
            )
            
            # Display results
            st.success("Prediction complete!")
            
            # Plot predictions
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(actual_prices, color='black', label="Actual Prices")
            ax.plot(predicted_prices, color='green', label="Predicted Prices")
            ax.set_title(f"{ticker} Stock Price Prediction")
            ax.set_xlabel("Time")
            ax.set_ylabel("Stock Price")
            ax.legend()
            st.pyplot(fig)
            
            # Plot training loss
            st.subheader("Training Progress")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title("Model Loss Progress")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Rishikesh")
>>>>>>> 79af6cf231f144b5e4ce36268d5e90a2804a8e65
