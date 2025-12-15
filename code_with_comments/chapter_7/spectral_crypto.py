# --- Import required libraries ---
from datetime import datetime
import time
from scipy.stats import linregress
import os
import requests
import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
import torch
from spectralnet import SpectralNet
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ===========================================================
# === 1. INITIAL SETUP AND COIN LIST DOWNLOAD FROM API ===
# ===========================================================

# Create a cache directory for saving API results locally
subdirectory = "cache"
os.makedirs(subdirectory, exist_ok=True)
random.seed(25032005)

# --- Define API parameters ---
vs_currency = 'usd'               # Target currency for prices
order = 'market_cap_desc'         # Sort by descending market capitalization
per_page = 250                    # Maximum number of coins per API request
total_coins = 1000                # Retrieve top 1000 coins
pages = total_coins // per_page   # Number of pages to iterate over

all_coins = []

# --- Download market data (coin IDs, symbols, names, prices, market cap) ---
for page in range(1, pages + 1):
    url = f'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': vs_currency,
        'order': order,
        'per_page': per_page,
        'page': page
    }
    response = requests.get(url, params=params)
    data = response.json()
    all_coins.extend(data)

# --- Store general coin info ---
df_coins = pd.DataFrame(all_coins, columns=['id', 'symbol', 'name', 'market_cap', 'current_price'])

# ===========================================================
# === 2. HISTORICAL DATA FETCHING AND CACHING PER COIN ===
# ===========================================================

def get_historical_data(crypto_id, vs_currency, from_date, to_date):
    """
    Download historical price, market cap, and volume data for a cryptocurrency
    between given dates from CoinGecko.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart/range"
    params = {
        'vs_currency': vs_currency,
        'from': str(int(from_date.timestamp())),
        'to': str(int(to_date.timestamp())),
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'prices' not in data:
        print("Error:", data)
        return None

    # Convert API response into a structured DataFrame
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Add market cap and volume data if available
    if 'market_caps' in data:
        df['market_cap'] = [item[1] for item in data['market_caps']]
    if 'total_volumes' in data:
        df['volume'] = [item[1] for item in data['total_volumes']]

    return df

# --- Set the date range for analysis ---
vs_currency = 'usd'
from_date = datetime(2024, 6, 1)
to_date = datetime(2024, 12, 31)

# Lists for computed indicators
Std_of_Price = []
Std_of_Market_Cap = []
Std_of_Volume = []
Log1p_Price = []
Log1p_Market_Cap = []
linreg_slope_price = []
ommited_coins = []

# --- Loop through each cryptocurrency and fetch data ---
for crypto_id in df_coins.id:
    print(crypto_id)

    # Try to load from local cache first to save API calls
    if os.path.exists(f'cache/{crypto_id}.pkl'):
        with open(f'cache/{crypto_id}.pkl', 'rb') as f:
            df = pickle.load(f)
        from_cache = True
    else:
        df = None
        from_cache = False
        # Retry until data is successfully retrieved
        while df is None:
            df = get_historical_data(crypto_id, vs_currency, from_date, to_date)
            if df is not None:
                with open(f'cache/{crypto_id}.pkl', 'wb') as f:
                    pickle.dump(df, f)
            else:
                time.sleep(100)  # Wait to avoid API rate limits

    # Skip if no valid data
    if df.empty:
        ommited_coins.append(crypto_id)
        continue

    # Skip coins with zero-valued time series (invalid data)
    if df['price'].mean() == 0 or df['market_cap'].mean() == 0 or df['volume'].mean() == 0:
        ommited_coins.append(crypto_id)
        continue

    # --- Compute normalized variability and trend features ---
    std_price = df['price'].std() / df['price'].mean()
    std_market_cap = df['market_cap'].std() / df['market_cap'].mean()
    std_volume = df['volume'].std() / df['volume'].mean()
    log1p_price = np.log1p(df['price'].mean())
    log1p_market_cap = np.log1p(df['market_cap'].mean())

    # Linear regression slope to capture trend direction
    slope, _, _, _, _ = linregress(range(len(df['price'])), df['price'])

    # Store computed statistics
    Std_of_Price.append(std_price)
    Std_of_Market_Cap.append(std_market_cap)
    Std_of_Volume.append(std_volume)
    Log1p_Price.append(log1p_price)
    Log1p_Market_Cap.append(log1p_market_cap)
    linreg_slope_price.append(slope)

    # Create/Update DataFrame of all coin features
    coin_characteristics = pd.DataFrame({
        'Std_of_Price': Std_of_Price,
        'Std_of_Market_Cap': Std_of_Market_Cap,
        'Std_of_Volume': Std_of_Volume,
        'log1p_price': Log1p_Price,
        'log1p_market_cap': Log1p_Market_Cap,
        'linreg_slope_price': linreg_slope_price
    })

    # Pause slightly to prevent API rate limit issues
    if not from_cache:
        time.sleep(10)

# Assign valid coin IDs as index
coin_characteristics.index = [cid for cid in df_coins.id if cid not in ommited_coins]

# ===========================================================
# === 3. DETERMINE CLUSTER COUNT USING EIGENVALUE GAP ======
# ===========================================================

# Construct a K-nearest-neighbor graph
n_neighbors = 10
adjacency_matrix = kneighbors_graph(coin_characteristics, n_neighbors, mode='connectivity').toarray()

# Compute Laplacian matrix (L = D - A)
degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
laplacian = degree_matrix - adjacency_matrix

# Compute the smallest eigenvalues of Laplacian
eigenvalues, _ = eigsh(laplacian, k=10, which='SM')

# --- Plot eigenvalue spectrum to find the "gap" indicating cluster count ---
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalue Spectrum (Gap Method for Cluster Count)")
plt.savefig("eigenvalue_gap.jpg")
plt.show()

# Based on the plot, we choose number of clusters manually
nr_clusters = 8

# ===========================================================
# === 4. SPECTRAL CLUSTERING USING SpectralNet MODEL =======
# ===========================================================

# Convert data to torch tensor for SpectralNet
X_t = torch.cat([torch.tensor(np.array(coin_characteristics), dtype=torch.float32)])

# --- Define SpectralNet configuration ---
spectralnet = SpectralNet(
    n_clusters=nr_clusters,
    should_use_ae=False,            # No autoencoder used
    should_use_siamese=False,       # No Siamese network
    spectral_batch_size=500,
    spectral_epochs=10,
    spectral_is_local_scale=False,
    spectral_n_nbg=20,
    spectral_scale_k=2,
    spectral_lr=1e-3,
    spectral_lr_decay=1e-1,
    spectral_hiddens=[128]*3 + [8], # Multi-layer hidden network
)

# --- Fit SpectralNet on the cryptocurrency features ---
spectralnet.fit(X_t)

# --- Predict clusters ---
predicted_clusters = spectralnet.predict(X_t)
print(predicted_clusters)

# ===========================================================
# === 5. VISUALIZE CLUSTERS AND FEATURE RELATIONSHIPS ======
# ===========================================================

# Simple 2D visualization of clusters
plt.figure(figsize=(6, 6))
plt.scatter(
    coin_characteristics.iloc[:, 2],     # Volume variability
    coin_characteristics.iloc[:, 5],     # Price slope
    c=predicted_clusters, cmap='viridis', s=50
)
plt.title('Predicted Cryptocurrency Clusters')
plt.xlabel('Std_of_Volume')
plt.ylabel('linreg_slope_price')
plt.savefig("coins_clusters.jpg")
plt.show()

# --- Pairwise feature relationships colored by cluster ---
variable_combinations = list(itertools.combinations(coin_characteristics.columns, 2))
num_combinations = len(variable_combinations)
num_cols = 4
num_rows = (num_combinations + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
axes = axes.flatten()

for i, (var_x, var_y) in enumerate(variable_combinations):
    ax = axes[i]
    sns.scatterplot(
        x=coin_characteristics[var_x],
        y=coin_characteristics[var_y],
        hue=predicted_clusters,
        palette="viridis",
        ax=ax,
        edgecolor='none',
        alpha=0.7
    )
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"{var_x} vs {var_y}")

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('Predicted_Clusters_all.jpg')
plt.show()

# ===========================================================
# === 6. CLUSTER ANALYSIS AND TIME SERIES VISUALIZATION ====
# ===========================================================

for i in range(8):
    print(f"Cluster {i+1} characteristics")
    print(coin_characteristics.index[predicted_clusters == i])

    ticker = coin_characteristics.index[predicted_clusters == i][0]
    print("min, max, mean, sd statistics for this cluster:")
    print(coin_characteristics.iloc[predicted_clusters == i, :].min())
    print(coin_characteristics.iloc[predicted_clusters == i, :].max())
    print(coin_characteristics.iloc[predicted_clusters == i, :].mean())
    print(coin_characteristics.iloc[predicted_clusters == i, :].std())

    # --- Plot time series for one representative coin in each cluster ---
    with open(f'cache/{ticker}.pkl', 'rb') as f:
        df = pickle.load(f)
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["price"], label='Historical Prices', color='blue')
    plt.title(f"Cluster {i+1} - Representative Coin: {ticker}")
    plt.xlabel("Date")
    plt.ylabel(f"{ticker} price in USD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cluster{i+1}_{ticker}_time_series.jpg")
    plt.show()
