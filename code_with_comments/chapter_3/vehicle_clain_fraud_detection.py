# --- Import necessary libraries ---
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re

# --- Kaggle API setup ---
# Set your Kaggle API credentials (you can create them from your Kaggle account)
os.environ['KAGGLE_USERNAME'] = "your_username"   # replace with your Kaggle username
os.environ['KAGGLE_KEY'] = "your_api_key"         # replace with your Kaggle API key

# --- Define dataset parameters ---
dataset_name = "shivamb/vehicle-claim-fraud-detection"
dataset_path = "vehicle-data"
dataset_file_name = "fraud_oracle.csv"

# --- Authenticate and download dataset ---
api = KaggleApi()
api.authenticate()

# Download the CSV file from Kaggle
api.dataset_download_file(dataset_name, file_name=dataset_file_name, path=dataset_path)

# --- Extract the downloaded ZIP file ---
with zipfile.ZipFile(f"{os.path.join(dataset_path, dataset_file_name)}.zip", 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

# --- Load dataset into pandas DataFrame ---
df = pd.read_csv(os.path.join(dataset_path, dataset_file_name))
print("Dataset Loaded Successfully!")
print(df.info())
pd.set_option('display.max_columns', None)
print(df.head(n=3))

# --- Remove columns not useful for prediction ---
columns_to_remove = [
    "PolicyNumber", "Days_Policy_Accident", "Days_Policy_Claim",
    "Deductible", "NumberOfCars", "AgeOfPolicyHolder", "Month", "WeekOfMonth",
    "DayOfWeek", "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Year"
]
df_cleaned = df.drop(columns=columns_to_remove)

# --- One-Hot Encoding for categorical variables ---
one_hot_cols = [
    "Sex", "AccidentArea", "MaritalStatus", "Fault", "PolicyType",
    "VehicleCategory", "AgentType", "BasePolicy", "AddressChange_Claim"
]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_one_hot = one_hot_encoder.fit_transform(df_cleaned[one_hot_cols])
encoded_one_hot_df = pd.DataFrame(encoded_one_hot, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

# Replace original categorical columns with encoded ones
df_cleaned = df_cleaned.drop(columns=one_hot_cols).reset_index(drop=True)
df_cleaned = pd.concat([df_cleaned, encoded_one_hot_df], axis=1)

# --- Encode binary variables as 0/1 ---
binary_cols = ["PoliceReportFiled", "WitnessPresent"]
for col in binary_cols:
    df_cleaned[col] = df_cleaned[col].map({'Yes': 1.0, 'No': 0.0})

# --- Convert VehiclePrice categorical ranges to numeric midpoints ---
df_cleaned["VehiclePrice"] = df_cleaned["VehiclePrice"].map({
    '20000 to 29000': 25000.0,
    '30000 to 39000': 35000.0,
    '40000 to 59000': 45000.0,
    'less than 20000': 10000.0,
    '60000 to 69000': 65000.0,
    'more than 69000': 85000.0
})

# --- Convert supplement and claim counts to numeric ---
df_cleaned["NumberOfSuppliments"] = df_cleaned["NumberOfSuppliments"].map({
    'none': 0.0,
    '1 to 2': 1.5,
    '3 to 5': 4.0,
    'more than 5': 8.0
})
df_cleaned["PastNumberOfClaims"] = df_cleaned["PastNumberOfClaims"].map({
    'none': 0.0,
    '1': 1.0,
    '2 to 4': 3.0,
    'more than 4': 5.0
})

# --- Convert vehicle age strings like "5 years" or "new" to numeric values ---
def extract_years(value):
    match = re.match(r"(\d+) years", str(value))
    return int(match.group(1)) if match else value

df_cleaned["AgeOfVehicle"] = (
    df_cleaned["AgeOfVehicle"]
    .replace("new", 0.0)
    .replace("more than 7", 10.0)
    .apply(extract_years)
)

# --- Encode categorical features ("Make" and "RepNumber") using embeddings ---
for embedable in ["Make", "RepNumber"]:
    label_encoder = LabelEncoder()
    df_cleaned[f"{embedable}_encoded"] = label_encoder.fit_transform(df[embedable])

    v_vocab_size = df_cleaned[f"{embedable}_encoded"].nunique()
    embedding_dim_v = int(np.sqrt(v_vocab_size))  # heuristic: sqrt of vocab size

    # Build embedding layer model
    v_input = Input(shape=(1,))
    v_embedding = Embedding(input_dim=v_vocab_size + 1, output_dim=embedding_dim_v)(v_input)
    v_embedding = Flatten()(v_embedding)

    v_model = Model(inputs=v_input, outputs=v_embedding)
    v_model.compile(optimizer="adam", loss="mse")

    # Train dummy embedding (no labels, just for initialization)
    dummy_target = np.zeros((df_cleaned.shape[0], embedding_dim_v))
    v_model.fit(df_cleaned[f"{embedable}_encoded"], dummy_target, epochs=5, batch_size=32, verbose=1)

    # Extract embeddings and merge with dataframe
    v_embeddings_model = Model(inputs=v_input, outputs=v_embedding)
    v_embeddings = v_embeddings_model.predict(df_cleaned[f"{embedable}_encoded"])
    v_embedding_df = pd.DataFrame(v_embeddings, columns=[f"{embedable}_emb_{i}" for i in range(embedding_dim_v)])

    df_cleaned = df_cleaned.drop(columns=[embedable, f"{embedable}_encoded"])
    df_cleaned = pd.concat([df_cleaned, v_embedding_df], axis=1)

print(df_cleaned.head(n=3))

# --- Separate features (X) and target (y) ---
y = df_cleaned["FraudFound_P"]
X = df_cleaned.drop(columns=["FraudFound_P"])

# --- Visualization helper: plot boxplots for feature distributions ---
def do_boxplot(title: str):
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=X, orient="h")
    plt.title("Boxplot of All Variables")
    plt.xlabel("Values")
    plt.ylabel("Variables")
    plt.savefig(title)
    plt.show()

# --- Visualize data before scaling ---
do_boxplot("discriminant_fraud_boxplot.png")

# --- Standardize numerical columns ---
scaler = StandardScaler()
X["VehiclePrice"] = scaler.fit_transform(X[["VehiclePrice"]])
do_boxplot("discriminant_fraud_boxplot_price_standardized.jpg")

X[["Age", "NumberOfSuppliments", "AgeOfVehicle"]] = scaler.fit_transform(
    X[["Age", "NumberOfSuppliments", "AgeOfVehicle"]]
)
do_boxplot("discriminant_fraud_boxplot_all_standardized.jpg")

# --- Split data into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Define and train a deep neural network (Keras model) ---
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

keras_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', 'recall', 'precision'])

# Class-weight to handle imbalance (fraud cases are fewer)
learning = keras_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight={0: 1, 1: 6},
    verbose=1
)

# --- Plot training curves for loss and accuracy ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(learning.history['loss'], label='Training Loss')
ax[0].plot(learning.history['val_loss'], label='Validation Loss', linestyle="dashed")
ax[0].set_title('Model Loss Over Epochs')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(learning.history['accuracy'], label='Training Accuracy')
ax[1].plot(learning.history['val_accuracy'], label='Validation Accuracy', linestyle="dashed")
ax[1].set_title('Model Accuracy Over Epochs')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()
plt.savefig("discriminant_fraud_learning_curve.jpg")
plt.show()

# --- Evaluate the neural network model ---
y_pred_keras = (keras_model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred_keras))

# --- Train and evaluate traditional ML models ---
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Ensemble Voting": VotingClassifier(estimators=[
        ("LR", LogisticRegression(max_iter=500)),
        ("RF", RandomForestClassifier()),
        ("GB", GradientBoostingClassifier())
    ], voting='soft')
}

# --- Evaluate and store metrics for each model ---
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# Add Keras model metrics for comparison
results["Keras Model"] = {
    "Accuracy": accuracy_score(y_test, y_pred_keras),
    "Precision": precision_score(y_test, y_pred_keras, zero_division=0),
    "Recall (Sensitivity)": recall_score(y_test, y_pred_keras),
    "F1 Score": f1_score(y_test, y_pred_keras),
    "ROC AUC": roc_auc_score(y_test, y_pred_keras),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_keras)
}

# --- Convert results to DataFrame for easy comparison ---
results_df = pd.DataFrame(results).T

# --- Visualization: model comparison bar chart ---
plt.figure(figsize=(12, 6))
results_df[["Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "ROC AUC"]].plot(kind='bar', figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Comparison of Model Performance on Fraud Detection")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.savefig("discriminant_fraud_comparison.jpg")
plt.show()

print(results_df)
