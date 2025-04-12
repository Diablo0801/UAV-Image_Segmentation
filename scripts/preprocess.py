import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define paths
train_image_dir = "data/train_images/"
train_label_dir = "data/train_labels/"
val_image_dir = "data/val_images/"
val_label_dir = "data/val_labels/"


# Function to load images and labels
def load_data(image_dir, label_dir, target_size=(256, 256)):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name)

        # Load image and label
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load image in RGB format
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load label as grayscale

        # Resize image and label to target size
        img = cv2.resize(img, target_size)
        label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor for labels

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


# Load training and validation data
train_images, train_labels = load_data(train_image_dir, train_label_dir)
val_images, val_labels = load_data(val_image_dir, val_label_dir)

# Normalize images to [0, 1] range
train_images = train_images / 255.0
val_images = val_images / 255.0

# Encode labels (if they are not already encoded as integers)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels.flatten()).reshape(train_labels.shape)
val_labels = label_encoder.transform(val_labels.flatten()).reshape(val_labels.shape)

# Save preprocessed data for later use
np.save("data/train_images.npy", train_images)
np.save("data/train_labels.npy", train_labels)
np.save("data/val_images.npy", val_images)
np.save("data/val_labels.npy", val_labels)

print("Data preprocessing complete!")