import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the saved model
model = tf.keras.models.load_model("models/l_net_best_laplacian.h5", compile=False)

# Define a color map for each class
# Example: Map class labels to BGR colors (OpenCV uses BGR format)
COLOR_MAP = {
    0: [0, 0, 0],  # Background (Black)
    1: [255, 0, 0],  # Building (Blue)
    2: [0, 255, 0],  # Road (Green)
    3: [0, 0, 255],  # Static Car (Red)
    4: [255, 255, 0],  # Tree (Cyan)
    5: [255, 0, 255],  # Low Vegetation (Magenta)
    6: [0, 255, 255],  # Human (Yellow)
    7: [128, 128, 128],  # Moving Car (Gray)
    8: [255, 255, 255]  # Background Clutter (White)
}


# Function to preprocess a single image
def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load image in RGB format
    original_size = img.shape[:2]  # Save original dimensions (height, width)
    img = cv2.resize(img, target_size)  # Resize to target size
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, original_size


# Function to preprocess a batch of images
def preprocess_batch(image_paths, target_size=(256, 256)):
    """Load and preprocess a batch of images for prediction."""
    images = []
    original_sizes = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load image in RGB format
        original_sizes.append(img.shape[:2])  # Save original dimensions (height, width)
        img = cv2.resize(img, target_size)  # Resize to target size
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images), original_sizes


# Function to resize the predicted mask to the original size
def resize_mask(mask, original_size):
    """Resize the predicted mask to the original image size."""
    return cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)


# Function to apply the color map to the mask
def apply_color_map(mask, color_map):
    """Convert a mask of class labels to an RGB image using a color map."""
    # Create an empty RGB image
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply the color map
    for class_label, color in color_map.items():
        rgb_mask[mask == class_label] = color

    return rgb_mask


# Function to save the predicted mask
def save_mask(mask, original_image_path, output_dir):
    """Save the predicted mask to the output directory with the same name as the input image."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the image name from the original path
    image_name = os.path.basename(original_image_path)

    # Construct the output path
    output_path = os.path.join(output_dir, image_name)

    # Save the mask
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to: {output_path}")


# Function to visualize the segmentation mask
def visualize_mask(mask, title="Segmentation Mask"):
    """Visualize the segmentation mask."""
    plt.imshow(mask)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Test the model on a single image
def test_single_image(image_path, output_dir):
    """Test the model on a single image and save the predicted mask."""
    # Preprocess the image
    img, original_size = preprocess_image(image_path)

    # Make a prediction
    pred_mask = model.predict(img)
    pred_mask = np.argmax(pred_mask, axis=-1)[0]  # Convert to class labels and remove batch dimension

    # Resize the predicted mask to the original size
    pred_mask_resized = resize_mask(pred_mask, original_size)

    # Apply the color map to the mask
    colored_mask = apply_color_map(pred_mask_resized, COLOR_MAP)

    # Save the predicted mask
    save_mask(colored_mask, image_path, output_dir)

    # Visualize the predicted mask
    visualize_mask(colored_mask, title="Predicted Mask")


# Test the model on a batch of images
def test_batch(image_paths, output_dir):
    """Test the model on a batch of images and save the predicted masks."""
    # Preprocess the batch of images
    images, original_sizes = preprocess_batch(image_paths)

    # Make predictions
    pred_masks = model.predict(images)
    pred_masks = np.argmax(pred_masks, axis=-1)  # Convert to class labels

    # Resize, color, and save the predicted masks
    for i, (mask, original_size, image_path) in enumerate(zip(pred_masks, original_sizes, image_paths)):
        mask_resized = resize_mask(mask, original_size)
        colored_mask = apply_color_map(mask_resized, COLOR_MAP)
        save_mask(colored_mask, image_path, output_dir)
        visualize_mask(colored_mask, title=f"Predicted Mask {i + 1}")


# Example usage
if __name__ == "__main__":
    # Define the output directory for saving masks
    output_dir = "output_masks"

    # Test on a single image
    test_single_image("000400.png", output_dir)