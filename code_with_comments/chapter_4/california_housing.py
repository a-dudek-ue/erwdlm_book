# --- Import necessary libraries ---
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt

# --- Load the California housing dataset ---
california_data = fetch_california_housing(as_frame=True)
X = california_data.data   # Features
y = california_data.target # Target variable (median house value)

# Drop spatial features to simplify the model
X = X.drop(columns=['Longitude', 'Latitude'])

print(f"Dataset shape: {X.shape}")

# --- Feature scaling ---
# Neural networks are sensitive to input feature scales.
scaler = StandardScaler()
X_s = pd.DataFrame(scaler.fit_transform(X))
X_s.columns = X.columns
X = X_s  # Replace original X with standardized version

# --- Define and train the neural network regressor ---
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(64, 64, 64),  # Three hidden layers, 64 neurons each
    activation='relu',                # ReLU activation
    solver='adam',                    # Adam optimizer
    max_iter=5000,                    # Training iterations
    random_state=25032005
)

mlp_regressor.fit(X.values, y)

# --- Compute permutation feature importance ---
# This measures how much model performance decreases when each feature is randomly shuffled.
perm_importance = permutation_importance(
    estimator=mlp_regressor,
    X=X.values,
    y=y,
    scoring="neg_mean_squared_error",  # Use MSE for regression
    n_repeats=10,                      # Repeat shuffling 10 times for stability
    random_state=25032005
)

print(perm_importance)

# --- Convert results to a DataFrame for readability ---
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# --- Plot permutation importance ---
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Permutation Importance (? MSE)")
plt.title("Permutation Feature Importance - MLP on California Housing")
plt.gca().invert_yaxis()  # Most important feature on top
plt.savefig("housing_vip_mlp_results.jpg")
plt.show()

# --- Define prediction function for SHAP (required by KernelExplainer) ---
def model_predict(X):
    return mlp_regressor.predict(X).flatten()

# --- Initialize SHAP Kernel Explainer ---
# KernelExplainer uses a sampling-based Shapley approximation suitable for black-box models.
explainer = shap.KernelExplainer(
    model_predict,
    shap.sample(X, 100),  # Use 100 background samples for efficiency
    random_state=25032005
)

# --- Compute SHAP values for the dataset ---
# SHAP values explain each feature’s contribution to the model prediction.
shap_values = explainer(X.values)

# --- Show SHAP value breakdowns for selected individual instances ---
for i in [3, 25, 2005]:
    print(f"************* SHAP values for instance {i}")
    print(shap_values[i, :])  # Numeric contributions for each feature

    # Visualize feature contributions for a single observation
    plt.figure(figsize=(18, 6))
    fig = shap.plots.bar(shap_values[i, :], show=False, show_data=True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"housing_shap_mlp_{i}_results.jpg")
    plt.show()

# --- Partial Dependence Plots (PDPs) ---
# These show how the predicted outcome changes with each feature,
# while keeping all other features constant (“ceteris paribus”).
# The plots are grouped in sets of 3 features for clarity.
for i in range(2):
    PartialDependenceDisplay.from_estimator(
        estimator=mlp_regressor,
        # Only plot PDPs for 200 randomly chosen samples to reduce clutter
        X=X.values[np.random.permutation(range(X.shape[0]))[:200],],
        features=list(range(X.shape[1]))[3*i:3*(i+1)],
        feature_names=california_data.feature_names,
        kind="both",  # Draw both individual and averaged dependence lines
        random_state=25032005
    )
    plt.savefig(f"housing_pdp_mlp_results{i}.jpg")
    plt.show()
