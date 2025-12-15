# --- Import required libraries ---
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import requests
from io import BytesIO
import os

# --- Create a directory to save output images ---
subdirectory = "Images"
os.makedirs(subdirectory, exist_ok=True)

# --- Load an image from a URL ---
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Kozyra_K_piramida.jpg/800px-Kozyra_K_piramida.jpg"
headers = {
    'User-Agent': 'MyCustomUserAgent/1.0'  # identify your request to avoid blocking
}
response = requests.get(image_url, headers=headers)
image = Image.open(BytesIO(response.content)).convert("RGB")  # ensure RGB format

# --- Generate superpixel segmentation using SLIC (Simple Linear Iterative Clustering) ---
# Segmentation divides the image into coherent regions (superpixels)
segmentation_fn = lime_image.SegmentationAlgorithm(
    'slic',          # SLIC algorithm for segmentation
    n_segments=40,   # number of superpixels
    compactness=25,  # balance color proximity vs. space proximity
    sigma=10         # smoothing parameter
)
segmentation = segmentation_fn(np.array(image))

# --- Visualize segmentation boundaries on the image ---
plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(np.array(image), segmentation, color=(0.5, 0, 0.5)))
plt.title("Superpixel Segmentation Borders")
plt.axis("off")
plt.savefig("Images/segments_piramid.png")
plt.show()

# --- Load pretrained ResNet-50 model for image classification ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # set model to inference mode (disable dropout, etc.)

# --- Define image preprocessing transformations ---
# These match the preprocessing used during ResNet-50 training (on ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to model input size
    transforms.ToTensor(),          # convert PIL image to tensor
    transforms.Normalize(           # normalize using ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Define a helper function to make batch predictions ---
def batch_predict(images):
    """
    Converts a list of RGB images into model predictions.
    Each image is transformed, stacked into a batch tensor,
    passed through ResNet50, and outputs softmax probabilities.
    """
    images = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
    with torch.no_grad():  # no gradient computation (faster inference)
        outputs = model(images)
        return torch.nn.functional.softmax(outputs, dim=1).numpy()

# --- Initialize LIME Image Explainer ---
explainer = lime_image.LimeImageExplainer()

# --- Generate LIME explanations for the image ---
# LIME perturbs superpixels and measures how changes affect model predictions.
explanation = explainer.explain_instance(
    np.array(image),          # input image (as numpy array)
    batch_predict,            # prediction function
    top_labels=100,           # analyze up to 100 top predicted labels
    hide_color=0,             # replace masked regions with black
    num_samples=10000,        # number of perturbed samples
    segmentation_fn=segmentation_fn,  # use pre-defined segmentation
    random_seed=25032005      # reproducibility
)

# --- Load ImageNet labels to map model outputs to class names ---
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = np.array(requests.get(LABELS_URL).text.splitlines())

# --- Loop through the top predicted labels and visualize explanations ---
for top_label in explanation.top_labels:
    print(labels[top_label])  # print class name
    if labels[top_label] in ["hen", "television", "Ibizan hound", "maze", "horse cart"]:
        # Get both masked (with and without other regions hidden)
        temp, mask = explanation.get_image_and_mask(
            top_label,
            num_features=3,   # top 3 important superpixels
            hide_rest=False   # keep full image visible
        )
        temp_b, mask_b = explanation.get_image_and_mask(
            top_label,
            num_features=3,
            hide_rest=True    # hide all but top 3 regions
        )
        predicted_label = labels[top_label]
        print(f"Predicted label: {predicted_label}")

        # --- Display both LIME explanation versions ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # (A) Highlighted regions with rest hidden
        axes[0].imshow(mark_boundaries(temp_b, mask_b, color=(0.9, 0, 0.9), mode='thick'))
        axes[0].set_title(f"LIME Explanation (masked): {predicted_label}")
        axes[0].axis("off")

        # (B) Highlighted regions over the original image
        axes[1].imshow(mark_boundaries(temp, mask, color=(0.9, 0, 0.9), mode='thick'))
        axes[1].set_title(f"LIME Explanation (overlay): {predicted_label}")
        axes[1].axis("off")

        # --- Save figure for this label ---
        plt.savefig(f"Images/Katarzyna_Kozyra_piramid_{predicted_label}.png")
        plt.show()
