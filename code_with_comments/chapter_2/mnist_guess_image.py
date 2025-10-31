# --- Import required libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random
import numpy as np
from PIL import Image

# Set a fixed random seed to ensure reproducibility of results
random.seed(25032005)

# --- Data preprocessing setup ---
# The MNIST dataset consists of grayscale (1-channel) images of handwritten digits (0–9).
# Each image is converted to a tensor and normalized using the standard MNIST mean and standard deviation.
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize with mean/std derived from the original MNIST dataset
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Load the MNIST datasets (training and test) ---
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# --- Create data loaders for efficient batch processing ---
# train_loader: used for training the model (shuffled for better generalization)
# test_loader: used for evaluating model accuracy (not shuffled)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# --- Define a simple Convolutional Neural Network (CNN) for digit recognition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer:
        # Input: 1 channel (grayscale image)
        # Output: 16 feature maps
        # Kernel size: 3x3, padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # Second convolutional layer:
        # Input: 16 channels, Output: 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Fully connected layer:
        # After two max pooling layers, the 28x28 image is reduced to 7x7 feature maps.
        # 32 feature maps * 7 * 7 = 1568 features going into the dense layer.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Output layer: 10 neurons, one per digit class (0–9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through the network layers
        
        # Apply first convolution, followed by ReLU activation
        x = F.relu(self.conv1(x))
        # Apply 2x2 max pooling (reduces spatial dimensions by half)
        x = F.max_pool2d(x, 2)

        # Apply second convolution + ReLU + another 2x2 max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten feature maps into a vector for the fully connected layer
        x = x.view(x.size(0), -1)

        # Pass through first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Output layer (raw logits, no activation here since CrossEntropyLoss expects logits)
        x = self.fc2(x)
        return x


# --- Set up training environment ---
# Choose GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance and move it to the chosen device
model = SimpleCNN().to(device)

# Define loss function (CrossEntropyLoss is standard for classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam is an adaptive gradient descent method)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
epochs = 5  # Number of full passes through the training data
model.train()  # Set the model in training mode (enables dropout, batchnorm, etc.)
for epoch in range(epochs):
    for data, target in train_loader:
        # Move both input data and labels to the correct device
        data, target = data.to(device), target.to(device)

        # Zero out gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: compute predictions
        output = model(data)

        # Compute the loss (how far predictions are from correct labels)
        loss = criterion(output, target)

        # Backward pass: compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

    # Print loss value for monitoring progress
    print(f"Epoch {epoch+1} complete. Last batch loss: {loss.item():.4f}")


# --- Inference: predict digits from a custom stitched MNIST image ---
predicted_sequence = []  # list to store predicted digits

# Path to the image file created earlier (contains 8 digits in one row)
image_path = "mnist_digit_samples.png"

# Open the stitched image using Pillow and convert to grayscale
stitched_img_pil = Image.open(image_path).convert("L")
print("PIL .size ->", stitched_img_pil.size)  # Expected output: (224, 28)

# Apply the same preprocessing (transform) as for training
stitched_image = transform(stitched_img_pil)

# Disable gradient computation since we’re only making predictions
with torch.no_grad():
    # Loop over the 8 digits in the stitched image (each 28px wide)
    for i in range(8):
        # Extract each 28x28 digit region from the stitched image
        single_digit_image = stitched_image[:, :, i * 28:(i + 1) * 28]

        # Add batch dimension (model expects [batch_size, 1, 28, 28])
        single_digit_image = single_digit_image.unsqueeze_(0)

        # Move image tensor to device
        single_digit_image = single_digit_image.to(device)

        # Run the CNN to get output scores (logits)
        output = model(single_digit_image)

        # Get the predicted class index (digit with the highest score)
        pred = output.argmax(dim=1).item()

        # Store the predicted digit in the sequence
        predicted_sequence.append(pred)

# Print the predicted digits from the stitched image
print("Predicted sequence:", predicted_sequence)
