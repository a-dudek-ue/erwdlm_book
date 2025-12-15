# ==============================================================
# STOCK PRICE FORECASTING – ECONOMETRIC vs DEEP LEARNING MODELS
# ==============================================================
# Models included:
#   • ARIMA (autoregressive integrated moving average)
#   • Holt-Winters Exponential Smoothing
#   • GARCH (volatility model)
#   • Brownian motion baseline
#   • GRU (Gated Recurrent Unit)
#   • LSTM (Long Short-Term Memory)
#
# Evaluation metrics: RMSE, MAE, MAPE
# ==============================================================
# Last updated: 2025-10
# ==============================================================

import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

# ==============================================================
# --- CONFIGURATION AND GLOBAL SETTINGS ------------------------
# ==============================================================

PRICE_COLUMN = 'Close'      # Column used for price forecasting
VALUE_COLUMN = "Return"     # Target variable (price or return)

PENALTY = -100              # Fallback value for failed forecasts
activation = 'relu'         # Neural network activation function
epochs = 250                # Number of training epochs
sequence_length = 8         # Time steps per training sequence
test_scaler = None          # Placeholder for test data scaler

random.seed(25032205)

# ==============================================================
# --- METRIC COMPUTATION FUNCTIONS -----------------------------
# ==============================================================

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (safe division)."""
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

# ==============================================================
# --- ECONOMETRIC FORECASTING MODELS ---------------------------
# ==============================================================

def forecast_arima(train_series, steps):
    """ARIMA(2,1,2) model forecasting."""
    model = ARIMA(train_series, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def forecast_holtwinters(train_series, steps):
    """Additive Holt-Winters Exponential Smoothing forecast."""
    model = ExponentialSmoothing(train_series, trend='add', seasonal=None)
    model_fit = model.fit(optimized=True)
    forecast = model_fit.forecast(steps)
    return forecast

def forecast_garch(train_series, steps):
    """Forecast conditional mean using GARCH(1,1)."""
    try:
        am = arch_model(train_series, p=1, q=1, mean='AR', lags=1, dist='normal')
        res = am.fit(disp='off')
        forecast = res.forecast(horizon=steps)
        forecast = forecast.mean.iloc[-1].values
    except:
        # If model fitting fails, use penalty vector
        forecast = [PENALTY] * steps
    return pd.Series(forecast, index=range(steps))

def forecast_brownian(train_series, steps, phi=0.8, noise_std=0.0):
    """Generate Brownian motion–style forecast (random walk with drift)."""
    forecast = []
    current_value = train_series.iloc[-1]
    for _ in range(steps):
        noise = np.random.normal(0, noise_std) if noise_std > 0 else 0
        current_value = phi * current_value + noise
        forecast.append(current_value)
    return pd.Series(forecast, index=range(len(train_series), len(train_series) + steps))

# ==============================================================
# --- LSTM FORECASTING MODEL -----------------------------------
# ==============================================================

def build_lstm(sequence_length, dropout_rate):
    """Initialize and compile a simple LSTM model."""
    global model_lstm
    if not model_lstm:
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, activation=activation, input_shape=(sequence_length, 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print(model_lstm.summary())

def forecast_lstm(train_series, steps, sequence_length=sequence_length, epochs=epochs):
    """Train LSTM on historical data and forecast future values."""
    data = train_series.values
    X, y = [], []

    # Build training sequences
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    dropout_rate = 0.1
    build_lstm(sequence_length, dropout_rate)
    model_lstm.fit(X, y, epochs=epochs, verbose=0)

    # Recursive forecasting
    forecast = []
    last_seq = data[-sequence_length:].reshape(1, sequence_length, 1)
    for _ in range(steps):
        pred = model_lstm.predict(last_seq, verbose=False)
        forecast.append(pred[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    return pd.Series(forecast)

# ==============================================================
# --- GRU FORECASTING MODEL ------------------------------------
# ==============================================================

def build_gru(sequence_length):
    """Initialize and compile a simple GRU model."""
    global model_gru
    if not model_gru:
        model_gru = Sequential()
        model_gru.add(GRU(50, activation=activation, input_shape=(sequence_length, 1)))
        model_gru.add(Dense(1))
        model_gru.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print(model_gru.summary())

def forecast_gru(train_series, steps, sequence_length=sequence_length, epochs=epochs):
    """Train GRU model and forecast next values recursively."""
    data = train_series.values
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    build_gru(sequence_length)

    model_gru.fit(X, y, epochs=epochs, verbose=0)
    forecast = []
    last_seq = data[-sequence_length:].reshape(1, sequence_length, 1)

    for _ in range(steps):
        pred = model_gru.predict(last_seq, verbose=False)
        forecast.append(pred[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    return pd.Series(forecast)

# ==============================================================
# --- PRICE RECONSTRUCTION FROM RETURNS ------------------------
# ==============================================================

def calculate_prices(strategy, forecast, scaler, last_known_price, retratio_factor, ind):
    """
    Rebuild actual price levels from forecasted values (depending on scaling strategy).
    Handles raw prices, scaled prices, or return ratios.
    """
    if forecast.isna().any() or forecast.isnull().any():
        forecast = pd.Series([PENALTY]*len(forecast), index=forecast.index)

    forecast.index = ind

    if strategy == "raw":
        return forecast

    if strategy in ["scaler", "scaler_retratios"]:
        forecast = pd.Series(scaler.inverse_transform(f.values.reshape(-1, 1))[:, 0])
        if strategy == "scaler":
            return forecast

    # Compute predicted prices from returns
    predicted_prices_from_returns = np.repeat(last_known_price, len(forecast)) * np.cumprod(1 + (forecast.values/retratio_factor))
    predicted_prices_from_returns = pd.Series(predicted_prices_from_returns, index=forecast.index)

    if predicted_prices_from_returns.isna().any() or predicted_prices_from_returns.isnull().any():
        predicted_prices_from_returns = pd.Series([PENALTY]*len(predicted_prices_from_returns), index=ind)

    return predicted_prices_from_returns

# ==============================================================
# --- MAIN EXPERIMENT LOOP -------------------------------------
# ==============================================================

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NFLX", "NVDA", "JPM", "BAC", "INTC"]

original_stdout = sys.stdout
subdirectory = "Images"
os.makedirs(subdirectory, exist_ok=True)
model_gru = None
model_lstm = None
random.seed(25032005)
steps = 7                     # Forecast horizon (days)
metrics_summary = []          # Store evaluation metrics
stock_cache = {}              # Cache stock data to avoid repeated downloads
lengths = [50,100,200,300,500,1000]  # Different training window lengths

# ==============================================================
# --- FORECASTING AND EVALUATION PIPELINE ----------------------
# ==============================================================

for length_of_history in reversed(lengths):
    for phase in range(0, 2):
        for ticker in tickers:
            original_stdout.write(f"Processing:  {ticker} {length_of_history} {phase} \n")

            # Retrieve cached or download new data
            if ticker in stock_cache.keys():
                data = stock_cache[ticker]
            else:
                data = None
                while data is None or data.empty:
                    data = yf.download(ticker, start='2020-01-01', end='2025-02-15')
                stock_cache[ticker] = data

            data = data.dropna()

            # Prepare columns and datasets
            data[VALUE_COLUMN] = data[PRICE_COLUMN]
            data_raw = data.copy()
            data_scaler = data.copy()
            data_retratios = data.copy()

            last_known_price = data[PRICE_COLUMN].iloc[-(steps + 1), 0]
            y_true = data[PRICE_COLUMN].iloc[-steps:, 0]

            # Train-test split for multiple strategies
            data_train_raw = data_raw.iloc[-(length_of_history + steps):-steps]
            data_test_raw = data_raw.iloc[-steps:]

            data_retratios[VALUE_COLUMN] = data_retratios[VALUE_COLUMN].pct_change()
            data_train_retratios = data_retratios.iloc[-(length_of_history + steps):-steps]
            data_test_retratios = data_retratios.iloc[-steps:]

            # Scale data for LSTM/GRU
            train_scaler = MinMaxScaler()
            data_train_scaler = pd.DataFrame(train_scaler.fit_transform(data_scaler.iloc[-(length_of_history + steps):-steps, [-1]]), columns=[VALUE_COLUMN])
            test_scaler = MinMaxScaler()
            data_test_scaler = pd.DataFrame(test_scaler.fit_transform(data_scaler.iloc[-steps:, [-1]]), columns=[VALUE_COLUMN])
            data_test_scaler = data_test_scaler[VALUE_COLUMN]

            # ------------------------------------------------------
            # PHASE 0: Neural Network Forecasts (LSTM / GRU)
            # ------------------------------------------------------
            if phase == 0:
                if length_of_history == 1000:
                    f = forecast_gru(data_train_scaler, steps)
                    f = forecast_lstm(data_train_scaler, steps)
            # ------------------------------------------------------
            # PHASE 1: Econometric and Deep Learning Models
            # ------------------------------------------------------
            else:
                f = forecast_arima(data_train_retratios, steps)
                arima_forecast = calculate_prices("retratios", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_retratios.index).to_period('D').to_timestamp())

                f = forecast_holtwinters(data_train_raw, steps)
                hw_forecast = calculate_prices("raw", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_raw.index).to_period('D').to_timestamp())

                f = forecast_garch(data_train_retratios, steps)
                garch_forecast = calculate_prices("retratios", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_retratios.index).to_period('D').to_timestamp())

                f = forecast_brownian(data_train_raw, steps)
                bm_forecast = calculate_prices("raw", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_raw.index).to_period('D').to_timestamp())

                f = forecast_gru(data_train_scaler, steps)
                gru_forecast = calculate_prices("scaler", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_scaler.index).to_period('D').to_timestamp())

                f = forecast_lstm(data_train_scaler, steps)
                lstm_forecast = calculate_prices("scaler", f, test_scaler, last_known_price, 1, pd.DatetimeIndex(data_test_scaler.index).to_period('D').to_timestamp())

                if phase == 1:
                    temp_file = open(f'Images/results_{sequence_length}_{epochs}_{activation}.txt', 'w', encoding="utf-8")
                    sys.stdout = temp_file

                # --------------------------------------------------
                # Combine all models and compute metrics
                # --------------------------------------------------
                mods = {
                    'ARIMA': arima_forecast,
                    'HoltWinters': hw_forecast,
                    'GARCH': garch_forecast,
                    'Brownian': bm_forecast,
                    'GRU': gru_forecast,
                    'LSTM': lstm_forecast
                }

                # Visualization grid
                fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(24, 18))
                idx = 0
                for model_name, forecast in mods.items():
                    y_pred = forecast.values
                    rmse = compute_rmse(y_true, y_pred)
                    mae = compute_mae(y_true, y_pred)
                    mape = compute_mape(y_true, y_pred)
                    metrics_summary.append({
                        'Ticker': ticker,
                        'History_length': length_of_history,
                        'Model': model_name,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    })

                    ax = axs[idx % 3][idx // 3]
                    idx += 1
                    ax.plot(data_raw[-(min(lengths[2], length_of_history) + steps):].index,
                            data_raw[-(min(lengths[2], length_of_history) + steps):][PRICE_COLUMN],
                            label='Historical Prices', color='blue')
                    ax.plot(data_raw[-steps:].index, forecast.values,
                            label=f'{model_name} Forecast', color='red', marker='o', linestyle='--')
                    ax.set_title(f"{ticker} ({model_name}) - Last {steps} Sessions")
                    ax.legend()
                    ax.grid(True)

                plt.tight_layout()
                plt.savefig(f"Images/iterative_{length_of_history}.jpg", bbox_inches='tight')
                plt.close()

# ==============================================================
# --- AGGREGATED PERFORMANCE ANALYSIS --------------------------
# ==============================================================

metrics_df = pd.DataFrame(metrics_summary)
print("\nEvaluation Metrics Summary:")
print(metrics_df)

# Average performance by model
overall_avg = metrics_df.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean().reset_index()
print(overall_avg)

# Global metric visualization
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
for i, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
    ax[i // 2, i % 2].bar(overall_avg['Model'], overall_avg[metric])
    ax[i // 2, i % 2].set_title(f"Average {metric}")
plt.tight_layout()
plt.savefig("Images/overall_average_comparison.png")
plt.close()

# Metrics by training window length
for length in lengths:
    length_avg = metrics_df.query(f"`History_length`=={length}").groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean().reset_index()
    print(f"Results for history length {length}:")
    print(length_avg)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
        ax[i // 2, i % 2].bar(length_avg['Model'], length_avg[metric])
        ax[i // 2, i % 2].set_title(f"Average {metric}")
    fig.suptitle(f"Forecast performance for History Length {length}")
    plt.tight_layout()
    plt.savefig(f"Images/iterative_length_{length}_average_comparison.jpg")
    plt.close()

# Metrics by ticker
for ticker in tickers:
    ticker_avg = metrics_df.query(f"`Ticker`=='{ticker}'").groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean().reset_index()
    print(f"Results for {ticker}:")
    print(ticker_avg)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
        ax[i // 2, i % 2].bar(ticker_avg['Model'], ticker_avg[metric])
        ax[i // 2, i % 2].set_title(f"Average {metric}")
    fig.suptitle(f"Forecast comparison for {ticker}")
    plt.tight_layout()
    plt.savefig(f"Images/iterative_ticker_{ticker}_average_comparison.jpg")
    plt.close()

# Restore stdout after redirecting output
temp_file.close()
sys.stdout = original_stdout
