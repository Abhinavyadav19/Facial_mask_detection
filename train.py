# =====================================================================
# TOPIC 51: FACE MASK DETECTION (CNN & TRANSFER LEARNING)
# Optimized for MobileNetV2 Pixel Scaling [-1, 1]
# =====================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. HYPERPARAMETERS
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 4
DATASET_DIR = "dataset"

# 2. LOAD DATASET (80% Train, 20% Validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. TRANSFER LEARNING PIPELINE
# Input tensor
inputs = Input(shape=(224, 224, 3))

# Scale pixels from [0, 255] to [-1, 1] for MobileNetV2
x = preprocess_input(inputs)

# Base model with frozen pre-trained weights
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

# 4. COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. TRAIN MODEL
print("\n>>> Re-training Face Mask Detector with Proper Preprocessing...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 6. CALCULATE EVALUATION METRICS
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)

print("\n================== FINAL METRICS ==================")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print("===================================================\n")

# 7. SAVE RE-TRAINED MODEL
model.save("mask_detector_model.keras")
print("High-accuracy model saved as mask_detector_model.keras!")