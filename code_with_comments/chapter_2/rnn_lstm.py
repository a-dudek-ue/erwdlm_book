import random
import numpy as np
import sys
import matplotlib.pyplot as plt

# --- Helper function: Console progress bar with current loss ---
def progress_bar(current, total, loss, bar_length=40):
    fraction = current / total  # fraction of completion
    arrow_length = int(fraction * bar_length)  # number of "=" characters
    arrow = '=' * arrow_length
    blue_arrow = f"\033[94m{arrow}\033[0m"  # blue-colored progress section
    spaces = ' ' * (bar_length - arrow_length)

    # Display formatted progress bar
    progress = f"\rProgress: [{blue_arrow}{spaces}] {fraction * 100:6.2f}% loss:{loss:3.2f}"
    sys.stdout.write(progress)
    sys.stdout.flush()


# --- Long Short-Term Memory (LSTM) implementation from scratch ---
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # input_size: number of input features
        # hidden_size: number of hidden units (memory cells)
        # output_size: number of output neurons
        # learning_rate: step size for gradient updates
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight matrices for the four LSTM gates:
        # Forget gate (Wf), Input gate (Wi), Candidate cell (Wc), Output gate (Wo)
        # Each gate receives both current input x_t and previous hidden state h_{t-1}
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01

        # Biases for each gate
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Output layer weights and bias
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))

    # --- Activation functions ---
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    # --- Forward pass ---
    def forward(self, x):
        T = len(x)  # number of time steps
        h = np.zeros((T, self.hidden_size, 1))  # hidden states
        c = np.zeros((T, self.hidden_size, 1))  # cell states
        y = np.zeros((T, self.output_size, 1))  # outputs

        # Initialize previous states to zeros
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))

        for t in range(T):
            # Concatenate previous hidden state and current input
            combined = np.vstack((h_prev, x[t]))

            # Gate computations
            ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)  # Forget gate
            it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)  # Input gate
            c_hat = self.tanh(np.dot(self.Wc, combined) + self.bc)  # Candidate cell state
            ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)  # Output gate

            # Update cell state and hidden state
            c[t] = ft * c_prev + it * c_hat
            h[t] = ot * self.tanh(c[t])

            # Compute output at this time step
            y[t] = np.dot(self.V, h[t]) + self.b_y

            # Pass current states to next time step
            h_prev = h[t]
            c_prev = c[t]

        return y, h, c

    # --- Backward pass (Backpropagation Through Time for LSTM) ---
    def backward(self, x, y_true, y_pred, h, c):
        T = len(x)

        # Initialize gradients to zero
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dV = np.zeros_like(self.V)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        # Loop backward through time steps
        for t in reversed(range(T)):
            if t == 0:
                h_prev = np.zeros((self.hidden_size, 1))
                c_prev = np.zeros((self.hidden_size, 1))
            else:
                h_prev = h[t - 1]
                c_prev = c[t - 1]

            # Output layer error
            dy = y_pred[t] - y_true[t]
            dV += np.dot(dy, h[t].T)
            db_y += dy

            # Backpropagate into hidden layer
            dh = np.dot(self.V.T, dy) + dh_next

            # Recompute gate activations for derivatives
            combined = np.vstack((h_prev, x[t]))

            # Output gate gradient
            Wo_activation = np.dot(self.Wo, combined) + self.bo
            ot = self.sigmoid(Wo_activation)
            do = dh * self.tanh(c[t]) * ot * (1 - ot)

            # Cell state gradient
            dc = dh * ot * (1 - np.tanh(c[t]) ** 2) + dc_next

            # Forget gate gradient
            Wf_activation = np.dot(self.Wf, combined) + self.bf
            ft = self.sigmoid(Wf_activation)
            dft = dc * c_prev * ft * (1 - ft)

            # Input and candidate gradients
            Wi_activation = np.dot(self.Wi, combined) + self.bi
            it = self.sigmoid(Wi_activation)
            Wc_activation = np.dot(self.Wc, combined) + self.bc
            c_hat = self.tanh(Wc_activation)
            dit = dc * c_hat * it * (1 - it)
            dc_hat = dc * it * (1 - np.tanh(Wc_activation) ** 2)

            # Accumulate gradients for weights and biases
            dWf += np.dot(dft, combined.T)
            dWi += np.dot(dit, combined.T)
            dWc += np.dot(dc_hat, combined.T)
            dWo += np.dot(do, combined.T)
            dbf += dft
            dbi += dit
            dbc += dc_hat
            dbo += do

            # Propagate gradient to previous time step
            dcombined = (np.dot(self.Wf.T, dft) +
                         np.dot(self.Wi.T, dit) +
                         np.dot(self.Wc.T, dc_hat) +
                         np.dot(self.Wo.T, do))
            dh_next = dcombined[:self.hidden_size, :]
            dc_next = dc * ft

        return dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dV, db_y

    # --- Gradient update step ---
    def update_parameters(self, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dV, db_y):
        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wc -= self.learning_rate * dWc
        self.Wo -= self.learning_rate * dWo
        self.bf -= self.learning_rate * dbf
        self.bi -= self.learning_rate * dbi
        self.bc -= self.learning_rate * dbc
        self.bo -= self.learning_rate * dbo
        self.V -= self.learning_rate * dV
        self.b_y -= self.learning_rate * db_y

    # --- Training loop ---
    def train(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_train, y_train):
                # Forward and backward passes
                y_pred, h, c = self.forward(x)
                loss = np.sum((y_pred - y_true) ** 2)  # mean squared error
                total_loss += loss

                # Compute and apply gradients
                grads = self.backward(x, y_true, y_pred, h, c)
                self.update_parameters(*grads)

                # Display training progress
                progress_bar(epoch, epochs, total_loss)

    # --- Predict future values given an input sequence ---
    def predict(self, x):
        y_pred, h, c = self.forward(x)
        return y_pred


# --- Generate synthetic training data: cos(x) sequence ---
random.seed(25032005)
x_train = np.linspace(0, 8 * np.pi, 400)
y_train = (np.cos(x_train)).reshape(-1, 1)

sequence_length = 23
x_seq, y_seq = [], []

# Create sliding window sequences for training
for i in range(len(x_train) - sequence_length):
    x_seq.append(y_train[i:i + sequence_length])
    y_seq.append(y_train[i + 1:i + sequence_length + 1])

# Reshape for (time_steps, input_size, batch_size=1)
x_seq = [x.reshape(-1, 1, 1) for x in x_seq]
y_seq = [y.reshape(-1, 1, 1) for y in y_seq]

# --- Initialize and train the LSTM ---
rnn = LSTM(input_size=1, hidden_size=10, output_size=1, learning_rate=0.02)
rnn.train(x_seq, y_seq, epochs=4000)

# --- Predict future cosine values beyond training range ---
x_future = np.linspace(8 * np.pi, 10 * np.pi, 40)
y_future = np.zeros_like(x_future).reshape(-1, 1, 1)

predictions = []
last_seq = y_train[-sequence_length:].reshape(-1, 1, 1)

# Autoregressive prediction loop
for _ in range(len(x_future)):
    y_pred, h, c = rnn.forward(last_seq)
    last_seq = np.roll(last_seq, -1, axis=0)
    last_seq[-1] = y_pred[-1]
    predictions.append(y_pred[-1].item())

# --- Plot results ---
plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, label="Training Data (cos(x))")
plt.plot(x_future, predictions, label="Predictions (8? to 10?)", linestyle='--')
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.title("LSTM trained to predict cosine wave")
plt.legend()
plt.savefig("cos2_lstm.jpg")
plt.show()
