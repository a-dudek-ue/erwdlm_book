# --- Import required libraries ---
import os
import tensorflow as tf
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from recommenders.utils.timer import Timer                     # Utility for measuring time
from recommenders.models.ncf.ncf_singlenode import NCF          # Neural Collaborative Filtering model
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.evaluation.python_evaluation import (         # Evaluation metrics
    map, ndcg_at_k, precision_at_k, recall_at_k
)
import requests

# ==========================================================
# === 1. DOWNLOAD BOOK RATINGS AND METADATA FROM GITHUB ===
# ==========================================================

# URLs for dataset CSV files (Book ratings and metadata)
urls = {
    "BX-Book-Ratings.csv": "https://github.com/rochitasundar/Collaborative-Filtering-Book-Recommendation-System/raw/refs/heads/master/BX-Book-Ratings.csv",
    "BX-Books.csv": "https://github.com/rochitasundar/Collaborative-Filtering-Book-Recommendation-System/raw/refs/heads/master/BX-Books.csv"
}

# Directory to save the dataset files
save_dir = "data/recommender"
os.makedirs(save_dir, exist_ok=True)

# --- Helper function for downloading files ---
def download_file(url, save_path):
    """Download a file from a given URL and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download {url}")

# --- Download the datasets ---
for file_name, url in urls.items():
    file_path = os.path.join(save_dir, file_name)
    download_file(url, file_path)

print("Download complete.")

# =====================================================
# === 2. LOAD AND PREPARE THE DATA FOR RECOMMENDING ===
# =====================================================

# Load ratings dataset (users and their book ratings)
df_ratings = pd.read_csv(
    f"{save_dir}/BX-Book-Ratings.csv",
    delimiter=';',
    encoding='ISO-8859-1'
)

# Load book metadata (titles, authors, etc.)
df_books = pd.read_csv(
    f"{save_dir}/BX-Books.csv",
    delimiter=';',
    encoding='ISO-8859-1',
    on_bad_lines='skip'
)

# Display basic info about the datasets
print("Ratings shape:", df_ratings.shape)
print("Books shape:", df_books.shape)
print(df_books.head())

# --- Create a numeric book ID and merge datasets ---
df_books["Book-id"] = df_books.index
merged_df = pd.merge(df_ratings, df_books, "inner", "ISBN")

# Create a descriptive combined 'book' column (title + author)
merged_df["book"] = merged_df["Book-Title"] + " " + merged_df["Book-Author"]

# Select only relevant columns
recommender_data = merged_df[["User-ID", "Book-id", "book", "Book-Rating"]]
# Limit to a subset for faster computation (optional)
recommender_data = recommender_data.iloc[:12345, :]
print("Recommender data:", recommender_data.shape)

# =====================================================
# === 3. SYSTEM CONFIGURATION AND ENVIRONMENT INFO ===
# =====================================================

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("TensorFlow version: {}".format(tf.__version__))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# =====================================================
# === 4. DATA SPLITTING AND PREPROCESSING ============
# =====================================================

# Rename columns for consistency with NCF library
recommender_data.columns = ["userID", "itemID", "book", "rating"]

# Convert ratings to float type
recommender_data.loc[:, "rating"] = recommender_data["rating"].astype("float")

# --- Split data into train and test sets ---
SEED = 25032005
train, test = train_test_split(recommender_data, test_size=0.2, random_state=SEED)
train = train.sort_values(by="userID", ascending=True)
test = test.sort_values(by="userID", ascending=True)

# Ensure that all users and items in the test set exist in the training set
test = test[test["userID"].isin(train["userID"].unique())]
test = test[test["itemID"].isin(train["itemID"].unique())]

# Save data to CSV files for NCF input
train_file = "data/train.csv"
test_file = "data/test.csv"
recommender_data.to_csv(train_file, index=False)
recommender_data.to_csv(test_file, index=False)

# =====================================================
# === 5. BUILD AND TRAIN THE NCF RECOMMENDER MODEL ===
# =====================================================

# Prepare the dataset for the NCF model
data = NCFDataset(
    train_file=train_file,
    test_file=test_file,
    seed=SEED,
    col_user="userID",
    col_rating="rating",
    col_item="itemID"
)

# --- Define model hyperparameters ---
TOP_K = 10
EPOCHS = 150
BATCH_SIZE = 256

# --- Initialize the NCF (Neural Collaborative Filtering) model ---
# This uses a NeuMF hybrid architecture combining GMF + MLP
model = NCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",       # Neural Matrix Factorization
    n_factors=4,              # latent factors
    layer_sizes=[16, 8, 4],   # hidden layers
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)

# --- Train the NCF model ---
with Timer() as train_time:
    model.fit(data)
print(f"Took {train_time} seconds for training.")

# Save trained model for later reuse
model.save('data/model_serialized.tensorflow')

# =====================================================
# === 6. GENERATE PREDICTIONS ON THE TEST SET ========
# =====================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

with Timer() as test_time:
    users, items, preds = [], [], []
    item = list(test.itemID.unique())

    # Generate predictions for all user-item pairs
    for user in test.userID.unique():
        user_list = [user] * len(item)
        users.extend(user_list)
        items.extend(item)
        preds.extend(list(model.predict(user_list, item, is_list=True)))

    # Combine into a DataFrame of predictions
    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

    # Merge with train data to filter out already-rated books
    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1).drop('book', axis=1)

print(f"Took {test_time} seconds for prediction.")
# Rescale predictions to a 1–10 scale
all_predictions["prediction"] = all_predictions["prediction"] * 10

# =====================================================
# === 7. MODEL EVALUATION (MAP, NDCG, PRECISION, RECALL)
# =====================================================

eval_map = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

# Print all metrics
print(
    "MAP:\t%f" % eval_map,
    "NDCG:\t%f" % eval_ndcg,
    "Precision@K:\t%f" % eval_precision,
    "Recall@K:\t%f" % eval_recall,
    sep='\n'
)

# =====================================================
# === 8. GENERATE PERSONALIZED BOOK RECOMMENDATIONS ===
# =====================================================

# Select a few example users to get recommendations for
users_to_make_recommendations = [list(train.userID.unique())[i] for i in [25, 3, 5]]

for i, user in enumerate(users_to_make_recommendations):
    print(f"\nUser ID: {user}")

    # Show top-rated books by the user
    user_ratings = recommender_data[recommender_data["userID"] == user]
    print("Books rated by user:")
    print(user_ratings[["book", "rating"]].sort_values(by="rating", ascending=False).head(10))

    user_ratings.to_csv(f"data/user_ratings_{i}.csv", index=False)

    # --- Recommend books the user hasn't rated yet ---
    rated_books = set(user_ratings["itemID"])
    unrated_books = list(set(recommender_data["itemID"]) - rated_books)

    # Predict user preferences for unrated books
    user_predictions = model.predict([user] * len(unrated_books), unrated_books, is_list=True)
    predictions_df = pd.DataFrame({"itemID": unrated_books, "prediction": user_predictions})
    predictions_df.prediction = predictions_df.prediction * 10

    # Merge predictions with book metadata for titles/authors
    predictions_df = predictions_df.merge(df_books, left_on="itemID", right_on="Book-id")

    # Select top 10 recommendations
    top_10_books = predictions_df.sort_values(by="prediction", ascending=False).head(10)
    print("\nTop 10 recommended books for user:")
    print(top_10_books[["Book-Title", "Book-Author", "prediction"]])

    # Save top recommendations to file
    top_10_books.to_csv(f"data/top_10_books_{i}.csv", index=False)
