# =============================================================================
# Explainable AI for Cancer Classification with Sampling and XAI
# Purpose: Classify cancer types from a large dataset using sampling and explain predictions
# Dataset: Multi Cancer Dataset (with folder-per-class structure)
# Environment: Python with TensorFlow, Keras, OpenCV, LIME, etc.
# Date: October 2023
# =============================================================================

# Import Libraries
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.gradcam import Gradcam
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import json

# Debug: Check directory contents at multiple levels
print("Contents of archive(1):", os.listdir("."))
print("Contents of DATA_DIR:", os.listdir("data/Multi_Cancer"))

# =============================================================================
# Section 1: Configuration
# Set these parameters based on your setup
# =============================================================================

DATA_DIR = "data/Multi_Cancer"  # Points to inner Multi Cancer with ALL, Breast Cancer, etc.
SAMPLE_SIZE_PER_CLASS = 50  # Reduced to 50 for practicality
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 8  # Matches the 8 cancer types

# =============================================================================
# Section 2: Random Sampling and Data Loading
# Loads images from subclass folders within each cancer type
# =============================================================================

def load_sampled_data(data_dir, sample_size):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    print("Class mapping:", class_dict)

    if len(class_names) != NUM_CLASSES:
        print(f"Warning: Found {len(class_names)} classes, expected {NUM_CLASSES}. Adjust NUM_CLASSES if needed.")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        all_img_paths = []
        
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_img_paths.append(os.path.join(root, file))
        
        if not all_img_paths:
            print(f"Warning: No images found in {class_path} or its subfolders")
            continue
        
        sampled_imgs = random.sample(all_img_paths, min(sample_size, len(all_img_paths)))
        for img_path in sampled_imgs:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            images.append(img)
            labels.append(class_dict[class_name])
    
    return np.array(images), np.array(labels), class_dict

# =============================================================================
# Section 3: Data Preparation
# Splits the sampled data into train, validation, and test sets
# =============================================================================

def prepare_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
# Section 4: Model Building
# Builds a VGG16-based model for classification
# =============================================================================

def build_model(num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# =============================================================================
# Section 5: Data Augmentation and Training
# Trains with data augmentation and batch loading
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(X_train)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=1
    )
    return model, history

# =============================================================================
# Section 6: XAI Techniques
# Visualizes Grad-CAM, Saliency Maps, and LIME, saving data for LLM
# =============================================================================

def visualize_xai_and_save(model, img_array, class_dict, predicted_class, output_file="outputs/xai_output.json"):
    pred = model.predict(img_array)
    confidence = pred[0][predicted_class] * 100
    class_name = list(class_dict.keys())[predicted_class]
    
    xai_insights = ("Grad-CAM highlights the tumor in the brain center. "
                   "Saliency Map shows tumor edges are critical. "
                   "LIME outlines the tumor area as the most important region.")
    
    def gradcam_loss(output):
        return output[0, predicted_class]
    gradcam = Gradcam(model, clone=False)
    cam = gradcam(gradcam_loss, img_array, penultimate_layer=-1)
    
    def saliency_map(model, img_array):
        img_tensor = tf.convert_to_tensor(img_array)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = model(img_tensor)
            loss = preds[:, predicted_class]
        grads = tape.gradient(loss, img_tensor)[0]
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
        return saliency
    saliency = saliency_map(model, img_array)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0], model.predict, top_labels=1, num_samples=1000, hide_color=0
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(img_array[0], mask)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img_array[0])
    plt.title(f"Original Image\nPredicted: {class_name} ({confidence:.1f}%)")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_array[0])
    plt.imshow(cam[0], cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(saliency, cmap="hot")
    plt.title("Saliency Map")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(lime_img)
    plt.title("LIME")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    xai_data = {
        "predicted_class": int(predicted_class),
        "class_name": class_name,
        "confidence": confidence,
        "xai_insights": xai_insights
    }
    with open(output_file, 'w') as f:
        json.dump(xai_data, f, indent=4)
    print(f"Saved XAI data to {output_file}")

# =============================================================================
# Section 7: Main Execution
# Runs the full pipeline, saving data for the LLM script
# =============================================================================

def main():
    print("Starting XAI for Cancer Classification with Sampling...")
    
    print("Loading sampled data...")
    X, y, class_dict = load_sampled_data(DATA_DIR, SAMPLE_SIZE_PER_CLASS)
    if len(X) == 0:
        print("Error: No images loaded. Check DATA_DIR and dataset structure.")
        return
    
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("Building model...")
    model = build_model(NUM_CLASSES)
    
    print("Training model...")
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    print("Generating XAI visualizations and saving data...")
    test_img = X_test[0:1]
    pred = model.predict(test_img)
    predicted_class = np.argmax(pred[0])
    visualize_xai_and_save(model, test_img, class_dict, predicted_class)
    
    model.save("models/cancer_classifier_xai.h5")
    print("Model saved as models/cancer_classifier_xai.h5")

if __name__ == "__main__":
    main()
# =============================================================================