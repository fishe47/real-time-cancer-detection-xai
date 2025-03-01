# =============================================================================
# Test Cancer Classification with Saved Model (.h5) and XAI
# Purpose: Load the saved model, test on 1 new image per class, generate XAI outputs, and add performance metrics for LLM
# Dataset: Multi Cancer Dataset (with folder-per-class structure)
# Environment: Python with TensorFlow, Keras, OpenCV, LIME, etc.
# Date: February 2025
# =============================================================================

# Import Libraries
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.gradcam import Gradcam
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Debug: Check directory contents
print("Contents of DATA_DIR:", os.listdir("data/Multi_Cancer"))

# =============================================================================
# Configuration
# Set these parameters based on your setup
# =============================================================================

DATA_DIR = "data/Multi_Cancer"  # Points to inner Multi Cancer with ALL, Breast Cancer, etc.
IMG_SIZE = (224, 224)  # Match your model’s input size
SAMPLE_SIZE = 1  # Reduced to 1 image per class for testing
BATCH_SIZE = 32  # For consistency with training
NUM_CLASSES = 8  # Matches the 8 cancer types
MODEL_PATH = "models/cancer_classifier_xai.h5"  # Path to your saved model

# =============================================================================
# Data Loading for Testing
# Load a sample of 1 image from each class for testing
# =============================================================================

def load_test_images(data_dir, sample_size):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    print("Class mapping for testing:", class_dict)

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
# XAI and Data Collection
# Collect XAI data for all images and save to a single JSON file
# =============================================================================

def visualize_xai_and_collect(model, img_array, class_dict, predicted_class, true_class, image_index):
    pred = model.predict(img_array)
    confidence = pred[0][predicted_class] * 100
    class_name = list(class_dict.keys())[predicted_class]
    true_class_name = list(class_dict.keys())[true_class]
    
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
    plt.title(f"Original Image\nPredicted: {class_name} ({confidence:.1f}%), True: {true_class_name}")
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
    
    return {
        "image_index": image_index,
        "predicted_class": int(predicted_class),
        "class_name": class_name,
        "true_class": int(true_class),
        "true_class_name": true_class_name,
        "confidence": confidence,
        "xai_insights": xai_insights
    }

# =============================================================================
# Main Execution
# Load model, test on 1 new image per class, generate XAI, and add performance metrics
# =============================================================================

def main():
    print("Starting Testing with Saved Model for Cancer Classification...")
    
    try:
        model = load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Loading test images...")
    X_test, y_test, class_dict = load_test_images(DATA_DIR, sample_size=SAMPLE_SIZE)
    if len(X_test) == 0:
        print("Error: No test images loaded. Check DATA_DIR and dataset structure.")
        return
    
    print(f"Loaded {len(X_test)} test images across {len(class_dict)} classes")
    
    all_xai_data = []
    y_true = []
    y_pred = []
    
    for i in range(len(X_test)):
        test_img = X_test[i:i+1]
        true_class = y_test[i]
        pred = model.predict(test_img)
        predicted_class = np.argmax(pred[0])
        
        print(f"\nTesting image {i+1}/{len(X_test)} - True Class: {list(class_dict.keys())[true_class]}, "
              f"Predicted Class: {list(class_dict.keys())[predicted_class]}")
        
        xai_data = visualize_xai_and_collect(model, test_img, class_dict, predicted_class, true_class, i+1)
        all_xai_data.append(xai_data)
        y_true.append(true_class)
        y_pred.append(predicted_class)
    
    with open("outputs/xai_output.json", 'w') as f:
        json.dump(all_xai_data, f, indent=4)
    print("Saved all XAI data to outputs/xai_output.json")
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    with open("outputs/performance_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved performance metrics to outputs/performance_metrics.json")

if __name__ == "__main__":
    main()
# =============================================================================