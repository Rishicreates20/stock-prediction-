<<<<<<< HEAD
# stock_prediction_nvidia.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Collection
def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data, lookback=60):
    """
    Prepare data for LSTM model
    """
    # Use only 'Close' price
    dataset = data['Close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create time series dataset
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Step 3: Model Building
def build_lstm_model(input_shape):
    """
    Create LSTM model architecture
    """
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

# Step 4: Training and Evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    """
    Train model and evaluate performance
    """
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    
    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    return predicted_prices, actual_prices, mse, mae, r2, history

# Step 5: Visualization
def plot_results(actual, predicted, history):
    """
    Visualize training process and predictions
    """
    plt.figure(figsize=(16, 8))
    plt.plot(actual, color='black', label="Actual NVDA Price")
    plt.plot(predicted, color='green', label="Predicted NVDA Price")
    plt.title('NVIDIA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('NVDA Stock Price')
    plt.legend()
    plt.show()
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":  # Corrected line
    # Configuration
    TICKER = 'NVDA'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    LOOKBACK = 60
    TEST_SIZE = 0.2
    
    # Get data
    data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    # Preprocess data
    X, y, scaler = preprocess_data(data, LOOKBACK)
    
    # Split data
    split = int((1 - TEST_SIZE) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Train and evaluate
    predicted_prices, actual_prices, mse, mae, r2, history = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, scaler
    )
    
    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Visualize results
=======
# stock_prediction_nvidia.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Collection
def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data, lookback=60):
    """
    Prepare data for LSTM model
    """
    # Use only 'Close' price
    dataset = data['Close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create time series dataset
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Step 3: Model Building
def build_lstm_model(input_shape):
    """
    Create LSTM model architecture
    """
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

# Step 4: Training and Evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    """
    Train model and evaluate performance
    """
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    
    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    return predicted_prices, actual_prices, mse, mae, r2, history

# Step 5: Visualization
def plot_results(actual, predicted, history):
    """
    Visualize training process and predictions
    """
    plt.figure(figsize=(16, 8))
    plt.plot(actual, color='black', label="Actual NVDA Price")
    plt.plot(predicted, color='green', label="Predicted NVDA Price")
    plt.title('NVIDIA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('NVDA Stock Price')
    plt.legend()
    plt.show()
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":  # Corrected line
    # Configuration
    TICKER = 'NVDA'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    LOOKBACK = 60
    TEST_SIZE = 0.2
    
    # Get data
    data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    # Preprocess data
    X, y, scaler = preprocess_data(data, LOOKBACK)
    
    # Split data
    split = int((1 - TEST_SIZE) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Train and evaluate
    predicted_prices, actual_prices, mse, mae, r2, history = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, scaler
    )
    
    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Visualize results
>>>>>>> 79af6cf231f144b5e4ce36268d5e90a2804a8e65
    plot_results(actual_prices, predicted_prices, history)