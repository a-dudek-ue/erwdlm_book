import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

class MultiLayerPerceptron:
    # --- Activation Functions and Loss ---
    def sigmoid(self, x):
        # Sigmoid activation function: maps input to (0, 1)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of sigmoid: used for backpropagation
        return x * (1 - x)

    def mse_loss(self, y_true, y_pred):
        # Mean Squared Error (MSE) loss function
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_loss_derivative(self, y_true, y_pred):
        # Derivative of MSE loss with respect to predictions
        return 2 * (y_pred - y_true) / y_true.size

    # --- Initialization ---
    def __init__(self, input_size, hidden_layers):
        # input_size: number of input neurons
        # hidden_layers: number of hidden layers in the network
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.network = {}  # dictionary to hold weight matrices

    def initialize_weights(self, size):
        # Initialize all network weights with simple deterministic values
        # (for clarity, not for performance — normally random initialization is used)
        
        # Output layer weights
        weights = np.array(list(range(1, size + 1))) / size
        weights.shape = (size, 1)
        self.network[f'W{self.hidden_layers}'] = weights

        # Hidden layer weights
        for i in range(1, self.hidden_layers):
            weights = np.array(list(range(1, size + 1)) * size) / size
            weights.shape = (size, size)
            self.network[f'W{i}'] = weights

        # Input layer weights
        weights = np.array(list(range(1, size + 1)) * self.input_size) / size
        weights.shape = (self.input_size, size)
        self.network[f'W{0}'] = weights

    # --- Forward Propagation ---
    def forward_propagation(self, X):
        # Computes the forward pass through the network
        cache = {}
        L = X
        cache['L0'] = L  # store input layer
        
        # Loop through each layer: input › hidden › output
        for i in range(1, self.hidden_layers + 2):
            L = self.sigmoid(np.dot(L, self.network[f'W{i-1}']))
            cache[f'L{i}'] = L  # store layer outputs for backprop
        return cache

    # --- Backward Propagation ---
    def backward_propagation(self, cache, X, Y, learning_rate):
        # Compute gradients and update weights using backpropagation

        # Step 1: output layer error
        L = cache[f"L{self.hidden_layers + 1}"]
        L_error = self.mse_loss_derivative(Y, L)
        L_delta = L_error * self.sigmoid_derivative(L)

        # Step 2: update output layer weights
        self.network[f'W{self.hidden_layers}'] -= learning_rate * cache[f"L{self.hidden_layers}"].T.dot(L_delta)

        # Step 3: backpropagate through hidden layers
        for i in reversed(range(1, self.hidden_layers + 1)):
            L_error = L_delta.dot(self.network[f'W{i}'].T)
            L_delta = L_error * self.sigmoid_derivative(cache[f"L{i}"])
            self.network[f'W{i-1}'] -= learning_rate * cache[f"L{i-1}"].T.dot(L_delta)

    # --- Prediction ---
    def predict(self, input):
        # Predict output for a single input sample
        weighted_sum = self.forward_propagation(input)[f'L{self.hidden_layers + 1}']
        return weighted_sum

    # --- Training Loop ---
    def train(self, X, Y, learning_rate, epochs):
        # Initialize weights and train for a given number of epochs
        self.initialize_weights(len(X))
        self.all_weights = []  # store weights over epochs for visualization
        
        for epoch in range(epochs):
            # Forward pass
            cache = self.forward_propagation(X)
            # Backward pass (update weights)
            self.backward_propagation(cache, X, Y, learning_rate)
            # Store current weights snapshot
            self.all_weights.append(self.network.copy())

            # Print loss every 10 epochs for monitoring
            if epoch % 10 == 0:
                loss = self.mse_loss(Y, cache[f'L{self.hidden_layers + 1}'])
                print(f">>> Epoch {epoch}, Loss: {loss}")

    def predict_array(self, inputs):
        # Predict outputs for an array of input samples
        return [self.predict(inputs[i]) for i in range(len(inputs))]


# --- Training Example ---

# Input: 3 binary features › output based on custom rule (nonlinear)
training_inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Corresponding binary labels
training_labels = np.array([0, 1, 1, 1, 1, 1, 1, 0])
training_labels.shape = (8, 1)

# Create and train the multi-layer perceptron
mlp = MultiLayerPerceptron(input_size=3, hidden_layers=2)
mlp.train(training_inputs, training_labels, learning_rate=0.1, epochs=80000)

# --- Evaluation ---
predicted_labels = mlp.predict_array(training_inputs)
predicted_labels = [1 if x > 0.5 else 0 for x in predicted_labels]

accuracy = accuracy_score(training_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(training_labels, predicted_labels))

# --- Visualization of Weight Evolution ---

# Visualize input layer weight changes across epochs
l, m = mlp.all_weights[0]["W0"].shape
fig, ax = plt.subplots(l, m)
for i in range(l):
    for j in range(m):
        line = [a["W0"][i, j] for a in mlp.all_weights]
        ax[i, j].plot(line, label=f'W_{i}_{j}', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i + j])
        ax[i, j].legend()
fig.suptitle('Input weights evolution\nacross epochs (Non-separable classes - MLP)', fontsize=11)
plt.show()

# Visualize hidden layer weight changes across epochs
l, m = mlp.all_weights[1]["W1"].shape
fig, ax = plt.subplots(l, m)
for i in range(l):
    for j in range(m):
        line = [a["W1"][i, j] for a in mlp.all_weights]
        ax[i, j].plot(line, label=f'W_{i}_{j}', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i + j])
        ax[i, j].legend()
fig.suptitle('Hidden layer weights evolution\nacross epochs (Non-separable classes - MLP)', fontsize=11)
plt.show()
