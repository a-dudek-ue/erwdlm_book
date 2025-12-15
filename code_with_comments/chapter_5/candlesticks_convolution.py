# ==============================================================
# CANDLESTICK IMAGE-BASED STOCK MOVEMENT PREDICTION
# ==============================================================
# This script:
#  1. Downloads recent stock data from Yahoo Finance.
#  2. Generates candlestick chart images for each time window.
#  3. Converts these images into numerical tensors.
#  4. Trains CNN models to classify future price movement (up/down).
#  5. Tests on the most recent time window and computes performance metrics.
# ==============================================================

import sys
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import io
import pickle
import os
import requests
import random
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import time

# ==============================================================
# --- PARAMETERS AND GLOBAL SETTINGS ---------------------------
# ==============================================================

# Different analysis window lengths (number of days per chart)
analysis_window_lengths = [20, 10, 5]

min_series_len = 10             # Minimum required length of price series
EPOCHS = 500                    # Number of CNN training epochs
FOLDER = "data"                 # Directory to store downloaded files
DOWNLOAD_DATA = True            # Flag to control downloading tickers

# ==============================================================
# --- FUNCTION: DOWNLOAD STOCK DATA ----------------------------
# ==============================================================

def download_data(ticker) -> pd.DataFrame:
    """Download historical data for a given ticker using yfinance."""
    try:
        ticker = yf.Ticker(ticker)
        stock_data = ticker.history(period='max')
    except:
        return pd.DataFrame()

    # Limit data to specific date range
    try:
        stock_data = stock_data.iloc[
            stock_data.index.tz_convert(None) > datetime.strptime('2023-03-15', '%Y-%m-%d'),
            :
        ]
        stock_data = stock_data.iloc[
            stock_data.index.tz_convert(None) <= datetime.strptime('2025-03-15', '%Y-%m-%d'),
            :
        ]
    except:
        pass

    return stock_data

# ==============================================================
# --- FUNCTION: CREATE CANDLESTICK CHART IMAGE -----------------
# ==============================================================

def get_stock_image(stock_data, i, ticker):
    """
    Generate a candlestick chart for the selected window of stock data.
    Returns the image in memory (BytesIO buffer) without saving to disk.
    """

    # Define a custom mplfinance style for visual clarity
    custom_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        rc={
            'axes.edgecolor': 'none',
            'axes.facecolor': 'black',
            'figure.facecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'grid.color': 'black'
        },
        marketcolors=mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit', wick='inherit',
            volume='inherit'
        )
    )

    # Generate candlestick chart without axes or grid
    mpf.plot(
        stock_data,
        type='candle',
        style=custom_style,
        axisoff=True,
        ylabel='',
        ylabel_lower='',
        tight_layout=True,
        volume=True,
        returnfig=True
    )

    # Save the figure to a memory buffer instead of a file
    memory_buffer = io.BytesIO()
    plt.savefig(memory_buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()
    return memory_buffer

# ==============================================================
# --- FUNCTION: PREPARE MULTIPLE IMAGES ------------------------
# ==============================================================

def prepare_images(stock_data, analysis_window_length, last, ticker) -> list:
    """Generate multiple chart images for each rolling window."""
    images = [
        get_stock_image(stock_data.iloc[i:i + analysis_window_length, :], i, ticker)
        for i in range(0, len(stock_data) - last - 1)
    ]
    return images

# ==============================================================
# --- FUNCTION: PREPROCESS IMAGES -------------------------------
# ==============================================================

def preprocess_images(memory_buffers):
    """
    Convert chart images from memory into standardized numpy arrays:
      - Convert RGBA to RGB if necessary
      - Resize to 256x256
      - Normalize to float32
    """
    images = []
    for buffer in memory_buffers:
        buffer.seek(0)
        image = Image.open(buffer)

        # Ensure RGB format
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize to CNN input shape
        image = image.resize((256, 256))
        image = np.array(image, dtype=np.float32)
        images.append(image)
        buffer.close()

    return images

# ==============================================================
# --- MODEL DEFINITION -----------------------------------------
# ==============================================================

random.seed(25032005)
tf.keras.utils.set_random_seed(25032005)

def create_model():
    """Define a convolutional neural network for image-based classification."""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (10, 10), activation='relu'),
        MaxPooling2D(),
        Conv2D(8, (4, 4), activation='relu', padding="same"),
        MaxPooling2D(),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(2, activation="softmax")  # Binary classification: up / down
    ])
    model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

# Create one CNN for each analysis window size
all_models = {awl: create_model() for awl in analysis_window_lengths}

# Early stopping callback to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ==============================================================
# --- DOWNLOAD NASDAQ TICKERS LIST -----------------------------
# ==============================================================

url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
os.makedirs(FOLDER, exist_ok=True)
filepath = os.path.join(FOLDER, "nasdaqlisted.txt")

response = requests.get(url)
if response.status_code == 200:
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(response.text)
    print(f"File downloaded and saved to {filepath}")
else:
    print("Error downloading file:", response.status_code)

# Read ticker symbols
df = pd.read_csv(filepath, sep="|", engine="python", skipfooter=1)
print(df.shape)
print(df.head())

# Restrict to last 25 tickers for demonstration
df = df.iloc[-25:]

# ==============================================================
# --- MAIN TRAINING AND TESTING LOOP ---------------------------
# ==============================================================

y_test = []  # True binary labels for final step
x_test = []  # Last chart images to test prediction

start_time = time.time()

for iter, ticker in enumerate(df.Symbol):
    history_data = download_data(ticker)

    # Process only if enough data available
    if len(history_data) > analysis_window_lengths[0] + min_series_len:
        to_test_x = {awl: None for awl in analysis_window_lengths}

        # Train models for each window size
        for analysis_window_length in analysis_window_lengths:
            print(f"***** {ticker} {iter} window={analysis_window_length} *****")

            # Compute daily percentage changes for label generation
            Y = history_data['Close'].pct_change()[analysis_window_lengths[0] + 1:]

            # Generate rolling chart images (except last, used for testing)
            X = prepare_images(history_data, analysis_window_length, analysis_window_lengths[0], ticker)[:-1]

            end_time = time.time()
            print(f"Time before {ticker} - window {analysis_window_length}: {end_time - start_time:.2f} s")
            start_time = time.time()

            # Convert images to numeric arrays
            X_images = preprocess_images(X)

            # Normalize pixel values (0–1)
            X_images = [X_image / 255.0 for X_image in X_images]

            # Train CNN
            all_models[analysis_window_length].fit(
                np.array(X_images[:-1]),
                np.array([1 if y >= 0 else 0 for y in Y[:-1]]),
                epochs=EPOCHS,
                validation_split=0.2,
                batch_size=32,
                callbacks=[early_stopping]
            )

            # Store the most recent image for test prediction
            to_test_x[analysis_window_length] = X_images[-1]

            # Save test label from final actual movement
            if analysis_window_lengths[-1] == analysis_window_length:
                y_test.append(1 if Y.iloc[-1] >= 0 else 0)

        # Save test examples per ticker
        x_test.append(to_test_x)

# ==============================================================
# --- PREDICTION PHASE -----------------------------------------
# ==============================================================

y_pred = np.zeros(len(y_test))
r_pred = np.zeros((len(y_test), 2))
all_p = {}
all_r = {}

# Predict using each trained model and combine
for analysis_window_length in analysis_window_lengths:
    print(f"Prediction for window length {analysis_window_length}:")
    r = all_models[analysis_window_length].predict(
        np.array([x[analysis_window_length] for x in x_test]),
        batch_size=32
    )
    r_pred += r  # Sum probabilities from different models

    # Predicted classes (0=down, 1=up)
    p = np.argmax(r, axis=1)
    print(p)

    all_p[analysis_window_length] = p
    all_r[analysis_window_length] = r

# ==============================================================
# --- EVALUATION AND RESULTS SAVING ----------------------------
# ==============================================================

temp_file = open('results.txt', 'w', encoding="utf-8")
orig_out = sys.stdout
sys.stdout = temp_file

# Print raw and computed metrics
print(y_pred)
print(y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# Restore console output
temp_file.close()
sys.stdout = orig_out
