# --- Import required libraries ---
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import json
import urllib.request
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
import cv2
import os

# --- Create output directory for saving visualizations ---
subdirectory = "Images"
os.makedirs(subdirectory, exist_ok=True)

# --- Load pretrained VGG16 model with ImageNet weights ---
# VGG16 expects input images of size 224x224 and RGB format.
model = VGG16(weights='imagenet')

# --- Download an example image (Jerusalem’s Jaffa Gate) from Wikipedia ---
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Jaffa_Gate_Jerusalem_02.JPG/1280px-Jaffa_Gate_Jerusalem_02.JPG"
headers = {'User-Agent': 'MyCustomUserAgent/1.0'}  # identify client to avoid request blocking
response = requests.get(url, headers=headers)

# Convert image bytes into RGB image and numpy array
img = Image.open(BytesIO(response.content)).convert('RGB')
img_np = np.array(img)
original_size = img.size  # store original image size for later resizing

# -------------------------------------------------------------
# Define prediction function compatible with LIME
# -------------------------------------------------------------
def predict_fn(images):
    """
    Function used by LIME to get model predictions.
    - Accepts a list of images (as NumPy arrays).
    - Resizes each image to 224x224 (VGG16 input requirement).
    - Preprocesses images (mean subtraction, channel scaling).
    - Returns softmax predictions from the VGG16 model.
    """
    imgs = []
    for image in images:
        # Resize to 224x224 pixels
        image = cv2.resize(image, (224, 224))
        imgs.append(image)

    imgs = np.array(imgs)
    imgs = preprocess_input(imgs.astype(np.float32))  # apply VGG16 preprocessing
    preds = model.predict(imgs)  # run forward pass
    return preds

# --- Initialize LIME image explainer ---
explainer = lime_image.LimeImageExplainer()

# --- Create SLIC superpixel segmentation ---
# This divides the image into homogeneous regions that LIME can perturb.
segmentation_fn = lime_image.SegmentationAlgorithm(
    'slic',
    n_segments=40,   # number of superpixels
    compactness=30   # color-space vs. distance weighting
)
segmentation = segmentation_fn(np.array(img))

# --- Visualize segmentation borders on the original image ---
print("Mark on image")
plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(np.array(img), segmentation, color=(0.5, 0, 0.5)))
plt.title("Superpixel Segmentation Borders")
plt.axis("off")
plt.savefig("Jeruzalem_explanation_superpixels.png")

# --- Run LIME to explain the model’s predictions for this image ---
# LIME perturbs superpixels and observes the effect on model predictions
explanation = explainer.explain_instance(
    img_np,                   # input image as numpy array
    predict_fn,               # prediction function
    top_labels=10,            # number of top labels to explain
    hide_color=0,             # color for hidden regions (black)
    num_samples=10000,        # number of perturbed samples
    segmentation_fn=segmentation_fn,  # use the segmentation defined above
    random_seed=25032005      # ensure reproducibility
)

# --- Download ImageNet class index for label lookup ---
class_index_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
with urllib.request.urlopen(class_index_url) as url_response:
    class_idx = json.load(url_response)

# --- Process and visualize explanations for each found class ---
for found_idx in explanation.local_exp.keys():
    pattern = class_idx[str(found_idx)][1]  # get human-readable class name
    print("Explanation available for:", pattern)

    # Retrieve intercept (base probability) and superpixel contributions
    intercept = explanation.intercept[found_idx]
    superpixel_weights = explanation.local_exp[found_idx]

    # Sum the top 5 most positive superpixel contributions
    contribution = sum(sorted([v[1] for v in superpixel_weights], reverse=True)[:5])

    print(f"Base probability (Intercept) for class {pattern}: {intercept:.4f}")
    print(f"Top 5 superpixel contributions for class {pattern}: {contribution:.4f}")

    # --- Get visualization for the explained class ---
    temp, mask = explanation.get_image_and_mask(
        found_idx,
        positive_only=True,   # show only positively contributing regions
        num_features=5,       # show top 5 most influential superpixels
        hide_rest=False       # keep rest of image visible
    )

    # Resize LIME image to original resolution
    temp_resized = Image.fromarray(temp).resize(original_size, Image.BILINEAR)

    # Overlay LIME mask boundaries in magenta
    img_boundary = mark_boundaries(np.array(temp_resized) / 255.0, mask, color=(1, 0, 1), mode="thick")

    # --- Plot and save the explanation ---
    plt.figure()
    plt.imshow(img_boundary)
    plt.title(
        f"LIME Explanation for class '{pattern}'\n"
        f"Intercept: {intercept:.4f}\n"
        f"Top-5 superpixel contribution: {contribution:.4f}",
        fontsize=9
    )
    plt.axis('off')
    plt.savefig(f"Jeruzalem_explanation_{pattern}.png")
    plt.show()
