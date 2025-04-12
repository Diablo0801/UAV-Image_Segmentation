import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# Laplacian Pyramid Loss Implementation
def gaussian_kernel(size=5, sigma=1.0):
    """Creates a 2D Gaussian kernel for smoothing."""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    return tf.tensordot(g, g, axes=0)

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Applies Gaussian blur to an image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, tf.shape(image)[-1], 1])
    return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')

def laplacian_pyramid_loss(y_true, y_pred, levels=3, kernel_size=5, sigma=1.0):
    """Computes the Laplacian Pyramid Loss between y_true and y_pred."""
    loss = 0.0
    for _ in range(levels):
        # Ensure y_true and y_pred are 4D tensors
        if len(y_true.shape) == 3:
            y_true = tf.expand_dims(y_true, axis=-1)
        if len(y_pred.shape) == 3:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Compute Gaussian blur
        y_true_blur = gaussian_blur(y_true, kernel_size, sigma)
        y_pred_blur = gaussian_blur(y_pred, kernel_size, sigma)

        # Compute Laplacian (difference between original and blurred)
        y_true_laplacian = y_true - y_true_blur
        y_pred_laplacian = y_pred - y_pred_blur

        # Add L1 loss for this level
        loss += tf.reduce_mean(tf.abs(y_true_laplacian - y_pred_laplacian))

        # Downsample for the next level
        y_true = tf.image.resize(y_true, [tf.shape(y_true)[1] // 2, tf.shape(y_true)[2] // 2], method='bilinear')
        y_pred = tf.image.resize(y_pred, [tf.shape(y_pred)[1] // 2, tf.shape(y_pred)[2] // 2], method='bilinear')

    return loss

# Combined Loss Function
def combined_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    y_pred_mask = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.expand_dims(y_pred_mask, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred_mask = tf.cast(y_pred_mask, tf.float32)
    lp_loss = laplacian_pyramid_loss(y_true, y_pred_mask, levels=3)
    total_loss = ce_loss + 0.1 * lp_loss
    return total_loss

# L-Net Architecture
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def attention_gate(skip_connection, gating_signal, inter_channels):
    theta_x = layers.Conv2D(inter_channels, 1, strides=1)(skip_connection)
    phi_g = layers.Conv2D(inter_channels, 1, strides=1)(gating_signal)
    add = layers.Add()([theta_x, phi_g])
    act = layers.ReLU()(add)
    psi = layers.Conv2D(1, 1, activation='sigmoid')(act)
    return layers.Multiply()([skip_connection, psi])

def aspp_module(input_tensor, filters=256):
    shape = tf.keras.backend.int_shape(input_tensor)
    pool = layers.GlobalAveragePooling2D()(input_tensor)
    pool = layers.Reshape((1, 1, shape[-1]))(pool)
    pool = layers.Conv2D(filters, 1, activation='relu')(pool)
    pool = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation='bilinear')(pool)
    conv1 = layers.Conv2D(filters, 1, dilation_rate=1, padding='same', activation='relu')(input_tensor)
    conv6 = layers.Conv2D(filters, 3, dilation_rate=6, padding='same', activation='relu')(input_tensor)
    conv12 = layers.Conv2D(filters, 3, dilation_rate=12, padding='same', activation='relu')(input_tensor)
    conv18 = layers.Conv2D(filters, 3, dilation_rate=18, padding='same', activation='relu')(input_tensor)
    concat = layers.Concatenate()([pool, conv1, conv6, conv12, conv18])
    output = layers.Conv2D(filters, 1, padding='same', activation='relu')(concat)
    return output

def build_l_net(input_shape=(1024, 1024, 3), num_classes=8):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    e1 = residual_block(x, 64)
    p1 = layers.MaxPooling2D()(e1)
    e2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D()(e2)
    e3 = residual_block(p2, 256)
    p3 = layers.MaxPooling2D()(e3)
    e4 = residual_block(p3, 512)
    p4 = layers.MaxPooling2D()(e4)
    bottleneck = aspp_module(p4)
    d1 = layers.UpSampling2D()(bottleneck)
    a1 = attention_gate(e4, d1, 256)
    d1 = layers.Concatenate()([d1, a1])
    d1 = residual_block(d1, 512)
    d2 = layers.UpSampling2D()(d1)
    a2 = attention_gate(e3, d2, 128)
    d2 = layers.Concatenate()([d2, a2])
    d2 = residual_block(d2, 256)
    d3 = layers.UpSampling2D()(d2)
    a3 = attention_gate(e2, d3, 64)
    d3 = layers.Concatenate()([d3, a3])
    d3 = residual_block(d3, 128)
    d4 = layers.UpSampling2D()(d3)
    a4 = attention_gate(e1, d4, 32)
    d4 = layers.Concatenate()([d4, a4])
    d4 = residual_block(d4, 64)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d4)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
model = build_l_net()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_loss,
              metrics=['accuracy'])
model.summary()

BATCH_SIZE = 8
EPOCHS = 25

# Create callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("l_net_best_laplacian.h5", save_best_only=True),
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
]

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the trained model
model.save("model.h5")
print("Model saved successfully!")

# Plot training history
import matplotlib.pyplot as plt
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Evaluation section (using saved model and data)
import tensorflow as tf
train_images = np.load("data/train_images.npy")
train_labels = np.load("data/train_labels.npy")
val_images = np.load("data/val_images.npy")
val_labels = np.load("data/val_labels.npy")

# Compile the loaded model for evaluation
model = tf.keras.models.load_model("l_net_best_laplacian.h5", compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Training Accuracy: {train_accuracy:.4f}")

val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")