# --- Import required libraries ---
from itertools import combinations  # for all subset combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from math import factorial  # for weighting in Shapley formula

# --- Generate synthetic data ---
np.random.seed(20050325)
n_samples = 500
X1 = np.random.normal(size=n_samples)         # Random noise
X2 = np.random.uniform(1, 4, size=n_samples)  # Feature 2
X3 = np.random.uniform(1, 4, size=n_samples)  # Feature 3
# Target variable depends mainly on X2 and X3 with noise
y = X2 + 3 * X3 + np.random.normal(scale=0.1, size=n_samples)

# Combine into DataFrame
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

# --- Display basic dataset statistics ---
simple_stats = data.describe()
print("\nSimple Statistics:")
print(simple_stats)

# --- Prepare input (X) and target (y) ---
X = data[['X1', 'X2', 'X3']]
y = data['y']

# --- Train a simple neural network regressor (MLP) ---
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=20050325)
mlp.fit(X, y)

# --- Manual breakdown analysis (incremental feature contributions) ---
# Select one instance to explain
instance_nr = 4
instance = X.iloc[instance_nr - 1]  # instance to explain

# Baseline: mean model prediction across all samples
current = baseline = mlp.predict(X).mean()

# Dictionary to store step-by-step contributions
contributions = {}
XX = X.copy()

# Sequentially replace one feature’s column with the instance’s value
for feature in XX.columns:
    XX.loc[:, feature] = instance[feature]  # replace feature column with instance value
    modified_prediction = mlp.predict(XX).mean()  # new mean prediction
    contributions[feature] = modified_prediction - current  # difference = contribution
    current = modified_prediction  # update for next iteration

# --- Print breakdown results ---
print("\nBreakdown Contributions for Instance 4:")
print(f"Baseline (average prediction): {baseline}")
for feature, contribution in contributions.items():
    print(f"{feature}: {contribution}")
print(f"Final Prediction after applying all features: {current}")

# --- Define the setup for Shapley value computation ---
instance_nr = 4
instance = X.iloc[instance_nr - 1]
m = len(X.columns)
S = set(range(1, m + 1))  # index set of all features {1, 2, 3}

# --- Define the value function v(S): model output given a subset of features fixed ---
def value_function(subset, X, mlp, instance):
    """
    Given a subset of features, replace only those features in X with the instance’s values.
    Then, compute the model’s average prediction.
    """
    data = X.copy()
    for s in subset:
        data[X.columns[s - 1]] = instance[X.columns[s - 1]]
    val = mlp.predict(data).mean()
    print(f"Value for subset {subset}: {val}")
    return val

# --- Compute Shapley values for each feature ---
def shapley_values(S, v, X, mlp, instance):
    """
    Compute Shapley values for all features:
    ?_i = ?_T?S\{i} [ (|T|! * (n-|T|-1)! / n!) * (v(T?{i}) - v(T)) ]
    where:
      - n = number of features
      - T = subset of all features excluding i
      - v(T) = model value when features in T are used
    """
    n = len(S)
    shapley_vals = {f"X{i}": 0 for i in S}

    # Loop through each feature i
    for i in S:
        # Loop through all subset sizes of features excluding i
        for subset_size in range(n):
            # Generate all subsets T of size `subset_size` that do not include i
            for T in combinations(S - {i}, subset_size):
                T = set(T)
                # Compute combinatorial weight for subset T
                weight = factorial(len(T)) * factorial(n - len(T) - 1) / factorial(n)
                # Compute marginal contribution of feature i to subset T
                marginal_contribution = v(T | {i}, X, mlp, instance) - v(T, X, mlp, instance)
                # Accumulate weighted contribution
                shapley_vals[f"X{i}"] += weight * marginal_contribution

    return shapley_vals

# --- Calculate Shapley values for the selected instance ---
shapley_result = shapley_values(S, value_function, X, mlp, instance)

# --- Display Shapley values for each feature ---
print("\nShapley Values (Feature Contributions):")
for element, value in shapley_result.items():
    print(f"{element}: {value:.6f}")
