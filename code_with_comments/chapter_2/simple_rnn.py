import numpy as np
import matplotlib.pyplot as plt

# --- Define a simple Recurrent Neural Network (RNN) ---
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # input_size: number of features in the input
        # hidden_size: number of neurons in the hidden state
        # output_size: number of outputs
        # learning_rate: step size for parameter updates
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight matrices initialization (small random values)
        self.U = np.random.randn(hidden_size, input_size) * 0.01  # input › hidden
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden › hidden (recurrent)
        self.V = np.random.randn(output_size, hidden_size) * 0.01 # hidden › output

        # Bias vectors initialization
        self.b_h = np.zeros((hidden_size, 1))  # hidden bias
        self.b_y = np.zeros((output_size, 1))  # output bias

    def tanh(self, x):
        # Activation function: hyperbolic tangent
        # Outputs in range (-1, 1)
        return np.tanh(x)

    def forward(self, x):
        # Perform forward pass through the RNN over all time steps
        T = len(x)  # number of time steps in the sequence
        h = np.zeros((T + 1, self.hidden_size, 1))  # store hidden states
        y = np.zeros((T, self.output_size, 1))      # store outputs

        # Process the sequence step by step
        for t in range(T):
            # h[t] depends on current input x[t] and previous hidden state h[t-1]
            h[t] = self.tanh(np.dot(self.U, x[t]) + np.dot(self.W, h[t - 1]) + self.b_h)
            # Compute output for this time step
            y[t] = np.dot(self.V, h[t]) + self.b_y

        return y, h

    def backward(self, x, y_true, y_pred, h):
        # Backpropagation Through Time (BPTT)
        T = len(x)

        # Initialize gradient matrices with zeros
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))  # gradient from next time step

        # Iterate backward through time
        for t in reversed(range(T)):
            # Output error at time step t
            dy = y_pred[t] - y_true[t]
            # Gradients for output weights and bias
            dV += np.dot(dy, h[t].T)
            db_y += dy

            # Backpropagate error into hidden layer
            dh = np.dot(self.V.T, dy) + dh_next
            dtanh = (1 - h[t] ** 2) * dh  # derivative of tanh

            # Gradients for input and recurrent weights
            dU += np.dot(dtanh, x[t].T)
            dW += np.dot(dtanh, h[t - 1].T)
            db_h += dtanh

            # Propagate gradient backward to previous time step
            dh_next = np.dot(self.W.T, dtanh)

        return dU, dW, dV, db_h, db_y

    def update_parameters(self, dU, dW, dV, db_h, db_y):
        # Gradient descent update rule for all parameters
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.V -= self.learning_rate * dV
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y

    def train(self, x_train, y_train, epochs=100):
        # Train the RNN over multiple epochs
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_train, y_train):
                # Forward pass
                y_pred, h = self.forward(x)

                # Compute mean squared error loss
                loss = np.sum((y_pred - y_true) ** 2)
                total_loss += loss

                # Backward pass (compute gradients)
                dU, dW, dV, db_h, db_y = self.backward(x, y_true, y_pred, h)

                # Update parameters
                self.update_parameters(dU, dW, dV, db_h, db_y)

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    def predict(self, x, h_prev=None):
        # Predict future outputs given a sequence
        y_pred, h = self.forward(x)
        return y_pred


# --- Generate training data (cos^2(x)) ---
x_train = np.linspace(0, 4 * np.pi, 100)          # x values from 0 to 4?
y_train = (np.cos(x_train)**2).reshape(-1, 1)     # target: cos2(x)

sequence_length = 20  # number of time steps per sequence
x_seq = []
y_seq = []

# Create overlapping training sequences for RNN
for i in range(len(x_train) - sequence_length):
    x_seq.append(y_train[i:i + sequence_length])
    y_seq.append(y_train[i + 1:i + sequence_length + 1])

# Reshape data for RNN input format: (time_steps, input_size, batch=1)
x_seq = [x.reshape(-1, 1, 1) for x in x_seq]
y_seq = [y.reshape(-1, 1, 1) for y in y_seq]

# --- Initialize and train the RNN ---
rnn = SimpleRNN(input_size=1, hidden_size=50, output_size=1, learning_rate=0.01)
rnn.train(x_seq, y_seq, epochs=4000)

# --- Predict future values beyond training range ---
x_future = np.linspace(4 * np.pi, 5 * np.pi, 40)  # range for future prediction
y_future = np.zeros_like(x_future).reshape(-1, 1, 1)

h_prev = None
predictions = []

# Start from the last part of the training sequence
last_seq = y_train[-sequence_length:].reshape(-1, 1, 1)

for _ in range(len(x_future)):
    # Predict next value
    y_pred, h_prev = rnn.forward(last_seq)

    # Shift sequence by one time step and insert new prediction
    last_seq = np.roll(last_seq, -1, axis=0)
    last_seq[-1] = y_pred[-1]

    # Store prediction for plotting
    predictions.append(y_pred[-1].item())

# --- Plot results ---
plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, label="Training Data (cos2x)")
plt.plot(x_future, predictions, label="Predictions (4? to 5?)", linestyle='--')
plt.xlabel("x")
plt.ylabel("cos2(x)")
plt.title("RNN trained to predict squared cosine wave")
plt.legend()
plt.show()
