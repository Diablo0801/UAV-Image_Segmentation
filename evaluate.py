import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lnet import LNet
from data_utils import SegmentationDataset

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = LNet(num_classes=8).to(device)
model.load_state_dict(torch.load("l_net_best.pth"))
model.eval()  # Set the model to evaluation mode

# Define dataset and DataLoader for validation/test images
val_set = SegmentationDataset("data/val_images", "data/val_labels", target_size=(1024, 1024))
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Define evaluation function
def evaluate(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)  # Get model prediction

            # Convert output to class labels
            _, predicted = torch.max(output, 1)
            all_preds.append(predicted.cpu().numpy())  # Move to CPU and store
            all_labels.append(labels.cpu().numpy())  # Move to CPU and store

    return np.array(all_preds), np.array(all_labels)

# Run the evaluation
predictions, ground_truth = evaluate(model, val_loader)

# Visualize a few predictions
def plot_segmentation_results(image, gt_mask, pred_mask, num_classes=8):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_mask, cmap='jet', alpha=0.7)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Segmentation")
    plt.imshow(pred_mask, cmap='jet', alpha=0.7)
    plt.axis('off')

    plt.show()

# Pick an index for testing and visualization
index = 0  # Change this for different images

# Get the corresponding image and masks
image_path = val_set.image_paths[index]
gt_mask_path = val_set.label_paths[index]
image = cv2.imread(image_path)
gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

# Get the predicted mask
pred_mask = predictions[index][0]  # Shape (H, W)

# Visualize the results
plot_segmentation_results(image, gt_mask, pred_mask)