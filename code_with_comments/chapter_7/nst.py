# --- Import required libraries ---
import keras
from scipy.optimize import fmin_l_bfgs_b          # Optimization algorithm (L-BFGS-B)
from imageio import imwrite                       # To save generated images
import time
import tensorflow
tensorflow.compat.v1.disable_eager_execution()    # Disable eager execution for compatibility with legacy Keras backend graph ops
keras.__version__

# --- Load target and style reference images ---
from keras.preprocessing.image import load_img, img_to_array
target_image_path = 'ad.jpg'                      # Path to target (content) image
style_reference_image_path = 'transfer_style_reference_2.jpg'  # Path to style reference image

# --- Determine size and scaling ---
width, height = load_img(target_image_path).size
img_height = 400                                 # Set desired height of the generated image
img_width = int(width * img_height / height)     # Keep aspect ratio consistent

import numpy as np
from keras import backend as K
from keras.applications import vgg19
from matplotlib import pyplot as plt

# ============================================================
# === 1. IMAGE PREPROCESSING AND DEPROCESSING UTILITIES =======
# ============================================================

def preprocess_image(image_path):
    """Load and preprocess an image for VGG19 model."""
    img = load_img(image_path, target_size=(img_height, img_width))  # Resize image
    img = img_to_array(img)                                          # Convert to numpy array
    img = np.expand_dims(img, axis=0)                                # Add batch dimension
    img = vgg19.preprocess_input(img)                                # Apply VGG19 normalization
    return img

def deprocess_image(x):
    """Convert a processed image back to normal RGB format."""
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]                                                # Convert from BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')                           # Clip values and cast to integers
    return x

# ============================================================
# === 2. LOAD IMAGES INTO KERAS BACKEND TENSORS ==============
# ============================================================

target_image = K.constant(preprocess_image(target_image_path))       # Constant tensor for content image
style_reference_image = K.constant(preprocess_image(style_reference_image_path))  # Constant tensor for style image
combination_image = K.placeholder((1, img_height, img_width, 3))     # Placeholder for the generated (combination) image

# Combine all three into a single tensor (for feeding through VGG19 at once)
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# ============================================================
# === 3. LOAD THE VGG19 MODEL AND DEFINE LOSS FUNCTIONS ======
# ============================================================

# Load pretrained VGG19 (ImageNet weights, excluding top classification layers)
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

# --- Define the different loss functions used in style transfer ---

def content_loss(base, combination):
    """Content loss measures how different the content is between images."""
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    """Compute Gram matrix (feature correlations) of an activation map."""
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    """Style loss: compares feature correlations (Gram matrices) of style and generated image."""
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    """Total variation loss encourages spatial smoothness in the generated image."""
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :]
    )
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))

# ============================================================
# === 4. DEFINE TOTAL LOSS FUNCTION ==========================
# ============================================================

# Create a dictionary mapping layer names to outputs
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Select layers for content and style representation
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1', 'block5_conv1']

# Weight coefficients for loss components
total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 0.025

# Initialize total loss
loss = K.variable(0.0)

# --- Add content loss ---
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]        # Features from target image
combination_features = layer_features[2, :, :, :]         # Features from generated image
loss = loss + content_weight * content_loss(target_image_features, combination_features)

# --- Add style loss for multiple layers ---
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl        # Average across all selected style layers

# --- Add total variation regularization ---
loss += total_variation_weight * total_variation_loss(combination_image)

# ============================================================
# === 5. COMPUTE GRADIENTS AND DEFINE EVALUATOR CLASS ========
# ============================================================

# Compute gradients of loss with respect to the generated image
grads = K.gradients(loss, combination_image)[0]

# Define a callable function to fetch both loss and gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# Evaluator class is used to efficiently supply both loss and gradients
class Evaluator(object):
    """Helper class to compute loss and gradients separately for L-BFGS."""
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# ============================================================
# === 6. OPTIMIZATION LOOP USING L-BFGS =====================
# ============================================================

result_prefix = 'style_transfer_result'
iterations = 20

# Flatten the initial target image (starting point for optimization)
x = preprocess_image(target_image_path)
x = x.flatten()

# --- Run iterative optimization ---
for i in range(iterations):
    print('Starting iteration', i + 1)
    start_time = time.time()

    # Perform one L-BFGS-B optimization step
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)

    # Convert generated array back to image
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)

    # Save intermediate result image
    fname = result_prefix + '_at_iteration_%d.png' % i
    imwrite(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

# ============================================================
# === 7. DISPLAY RESULTS =====================================
# ============================================================

plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.title("Original Content Image")
plt.figure()
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.title("Style Reference Image")
plt.figure()
plt.imshow(img)
plt.title("Final Stylized Image")
plt.show()
