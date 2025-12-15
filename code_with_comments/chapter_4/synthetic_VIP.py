import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# --- Set random seed for reproducibility ---
np.random.seed(20050325)

# --- Generate synthetic data ---
n_samples = 500
# Random normal variables (independent noise)
X1 = np.random.normal(size=n_samples)
X2 = np.random.normal(size=n_samples)
# Uniform variable with stronger relationship to y
X3 = np.random.uniform(1, 4, size=n_samples)

# Target variable depends mostly on X3 with small Gaussian noise
y = 2 * X3 + np.random.normal(scale=0.1, size=n_samples)

# Combine features and target into a DataFrame
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

# --- Display descriptive statistics of generated data ---
simple_stats = data.describe()
print("\nSimple Statistics:")
print(simple_stats)

# --- Define input features (X) and target variable (y) ---
X = data[['X1', 'X2', 'X3']]
y = data['y']

# --- Create and train a simple Multi-Layer Perceptron Regressor ---
mlp = MLPRegressor(
    hidden_layer_sizes=(10,),  # one hidden layer with 10 neurons
    activation='relu',         # ReLU activation function
    max_iter=1000,             # number of training epochs
    random_state=20050325
)
mlp.fit(X, y)

# --- Compute baseline model loss (mean squared error × number of samples) ---
baseline_loss = mean_squared_error(y, mlp.predict(X)) * len(y)

# --- Initialize dictionaries to store permutation importance values ---
importance_differences = {}
importance_ratios = {}

# Number of permutation iterations to average over (define if not already set)
n_vip_iterations = 30

# --- Compute permutation feature importance ---
# For each feature, shuffle its values multiple times and measure the increase in model loss.
for i in range(n_vip_iterations):
    for feature in X.columns:
        # Create a copy of feature data to permute
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[feature].values)

        # Evaluate model on permuted data
        permuted_loss = mean_squared_error(y, mlp.predict(X_permuted)) * len(y)

        # Initialize sums during the first iteration
        if i == 0:
            current_importance_differences_sum = 0.0
            current_importance_ratio_sum = 0.0
        else:
            current_importance_differences_sum = importance_differences[feature]
            current_importance_ratio_sum = importance_ratios[feature]

        # Accumulate the difference and ratio of losses
        importance_differences[feature] = current_importance_differences_sum + (permuted_loss - baseline_loss)
        importance_ratios[feature] = current_importance_ratio_sum + (permuted_loss / baseline_loss)

# --- Average importance values across iterations ---
for feature in X.columns:
    importance_differences[feature] = importance_differences[feature] / n_vip_iterations
    importance_ratios[feature] = importance_ratios[feature] / n_vip_iterations

# --- Display the estimated feature importances ---
print("\nVariable Importance (Average Loss Difference):")
print(importance_differences)

print("\nVariable Importance (Average Loss Ratio):")
print(importance_ratios)

# --- Optional: visualize feature importance ---
plt.figure(figsize=(8, 5))
plt.bar(importance_differences.keys(), importance_differences.values(), color='skyblue')
plt.title('Feature Importance (Permutation - Loss Difference)')
plt.xlabel('Feature')
plt.ylabel('Average Increase in Loss')
plt.grid(axis='y')
plt.show()
