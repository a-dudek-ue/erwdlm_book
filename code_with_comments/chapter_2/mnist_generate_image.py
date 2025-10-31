import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import random
import numpy as np

# Set a fixed random seed for reproducibility (so results are the same every run)
random.seed(25032005)

# --- Define data preprocessing steps ---
# 1. Convert images to PyTorch tensors
# 2. Normalize pixel values with mean = 0.1307 and std = 0.3081 (standard MNIST normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Load the MNIST training dataset ---
# MNIST dataset contains 28x28 grayscale images of digits 0–9
train_dataset = datasets.MNIST(
    root='./data',          # directory where data will be downloaded/stored
    train=True,             # load the training set (not the test set)
    download=True,          # download dataset if not found locally
    transform=transform     # apply preprocessing transform defined above
)

# --- Specify which digits to extract from the dataset ---
digits_to_create = [2, 5, 0, 3, 2, 0, 0, 5]

# --- Helper function to get one random image of a specific digit ---
def get_mnist_digit_image(dataset, digit):
    # Find indices in the dataset where the label matches the desired digit
    indices = [i for i, (_, label) in enumerate(dataset) if label == digit]
    # Randomly select one of those indices
    chosen_index = random.choice(indices)
    # Retrieve the corresponding image and label
    img, label = dataset[chosen_index]
    return img  # return image tensor of shape [1, 28, 28] (1 channel, 28x28 pixels)

# --- Collect one sample image for each requested digit ---
digit_images = [get_mnist_digit_image(train_dataset, d) for d in digits_to_create]

# --- Concatenate selected digit images horizontally ---
# Images have shape [1, 28, 28]; we concatenate along the width dimension (dim=2)
stitched_image = torch.cat(digit_images, dim=2)  # resulting shape: [1, 28, total_width]

# --- Define image size and resolution for saving ---
width_px = 224      # desired output width in pixels
height_px = 28      # height corresponds to MNIST image height
dpi = 300           # resolution (dots per inch)

# Convert pixel dimensions to inches (required by Matplotlib)
width_in = width_px / dpi
height_in = height_px / dpi

# --- Create the figure ---
fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)

# Remove extra padding/margins and axis labels for a clean output
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.axis('off')  # hide axes and ticks

# --- Display the stitched image ---
plt.imshow(stitched_image.squeeze().numpy(), cmap='gray')

# --- Save the resulting image to a file ---
plt.savefig("mnist_digit_samples.png", dpi=dpi, bbox_inches='tight', pad_inches=0)

# --- Show the image in a window ---
plt.show()
