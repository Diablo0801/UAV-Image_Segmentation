import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = tf.keras.models.load_model("models/l_net_best_laplacian.h5", compile=False)

# Load preprocessed data
train_images = np.load("data/train_images.npy")
train_labels = np.load("data/train_labels.npy")
val_images = np.load("data/val_images.npy")
val_labels = np.load("data/val_labels.npy")

# Compile the model (required for evaluation)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Training Accuracy: {train_accuracy:.4f}")

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")