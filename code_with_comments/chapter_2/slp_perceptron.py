import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

class SingleLayerPerceptron:
    def __init__(self, input_size):
        # Initialize weights: the first element is the bias (value 1), the rest are feature weights
        # Weights are initialized based on the input size and normalized by input_size
        self.weights = np.array([input_size] + list(range(input_size))) / input_size
        # all_weights stores the history of weights (for visualization)
        self.all_weights = self.weights

    def input_extended(self, input):
        # Add a bias term (1) to the input vector
        # Example: [x1, x2] › [1, x1, x2]
        return np.append([1], input)

    def predict(self, input):
        # Compute the weighted sum of inputs and weights
        weighted_sum = np.dot(self.input_extended(input), self.weights)
        # Activation function (step function): outputs 1 if sum > 0, otherwise 0
        activation = 1 if weighted_sum > 0 else 0
        return activation

    def train(self, inputs, labels, learning_rate=0.1, epochs=100):
        # Train the perceptron using the perceptron learning rule
        for epoch in range(epochs):
            convergent = True  # flag indicating whether the model has converged
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = prediction - labels[i]  # difference between predicted and actual label
                if error:
                    convergent = False  # if there is any error, continue training
                # Update rule for perceptron weights:
                # w = w - ? * error * x
                self.weights -= learning_rate * error * self.input_extended(inputs[i])
            print(f">>>>>> Epoch {epoch} Error: {error}")
            # Save the current weights for visualization
            self.all_weights = np.vstack((self.all_weights, np.array(self.weights)))
            if convergent:
                # Stop training early if no errors occurred (model has converged)
                break

    def predict_array(self, inputs):
        # Predict outputs for an array of inputs
        return [self.predict(inputs[i]) for i in range(len(inputs))]


# Training data — logical AND function
training_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
training_labels = np.array([0, 0, 0, 1])  # expected output of AND logic
training_labels.shape = (4,1)  # reshape for consistency

# Create and train the perceptron
slp = SingleLayerPerceptron(input_size=2)
slp.train(training_inputs, training_labels, learning_rate=0.01, epochs=200)

# Make predictions and evaluate model performance
predicted_labels = slp.predict_array(training_inputs)
accuracy = accuracy_score(training_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(training_labels, predicted_labels))

# Visualize weight updates across epochs
fig, ax = plt.subplots(2, 2)
num_columns = slp.all_weights.shape[1]

for i in range(num_columns):
    ax[i//2, i%2].plot(
        slp.all_weights[:, i],
        label=f'W{i}',
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    )
    ax[i//2, i%2].legend()

# Hide the empty subplot (if number of weights isn’t a multiple of 4)
ax[1,1].set_visible(False)

fig.suptitle('Weights estimation in subsequent epochs (Separable classes)', fontsize=13)
plt.show()


# with the same class definitions

training_inputs = np.array([[0,0,0], [0,0,1], [0,1, 0], [0,1, 1],[1,0,0], [1,0,1], [1,1, 0], [1,1, 1]])
training_labels = np.array([0, 1, 1, 1,1,1,1,0])

slp = SingleLayerPerceptron(input_size=3)

slp.train(training_inputs, training_labels, learning_rate=0.01, epochs=300)

predicted_labels=slp.predict_array(training_inputs)
accuracy = accuracy_score(training_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(training_labels, predicted_labels))
# Create a figure and a set of subplots
fig, ax = plt.subplots(2,2)

num_columns = slp.all_weights.shape[1]
for i in range(num_columns):
    ax[i//2,i%2].plot(slp.all_weights[:, i], label=f'W {i}',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
    ax[i//2,i%2].legend()
fig.suptitle('Weights estimation in subsequent epochs (Non-separable classes)', fontsize=13)
plt.show()
