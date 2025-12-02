Overview
This project implements a deep learning–based system for automatic brain tumor detection and classification using MRI images. A fine-tuned VGG16 model is used to categorize images into four classes: glioma tumor, meningioma tumor, pituitary tumor, and no tumor.
The system includes a Flask web application that allows users to upload MRI scans and receive instant predictions along with Grad-CAM visual explanations.

Key Features
High-accuracy classification using a transfer-learning-based VGG16 model
Real-time prediction through an easy-to-use Flask web interface
Grad-CAM attention maps showing where the model is focusing
Class-balanced training and strong augmentation for better generalization
Additional details provided for each tumor type (description, severity, recommended medical action)

Workflow
1. Data Processing
MRI images are resized to 128×128 and normalized to the 0–1 range
Extensive augmentation (rotation, zoom, brightness, flipping) improves robustness
Class weights are computed to handle dataset imbalance

2. Model Architecture
Base model: VGG16, pretrained on ImageNet
Custom classification head added:
Flatten
Dense(256, ReLU) → Dropout(0.3)
Dense(128, ReLU) → Dropout(0.2)
Dense(4, Softmax)
Optimized using the Adam optimizer

3. Training Strategy

Phase 1 (10 epochs):
All VGG16 layers frozen
Only classification head trained
Helps stabilize weights and extract domain-specific features

Phase 2 (up to 30 epochs):
Last convolutional block (Block 5) unfrozen
Low learning rate (1e-5) used
Allows fine-tuning for MRI-specific patterns

4. Explainability – Grad-CAM
Extracts gradients from the last convolutional layer (block5_conv3)
Constructs a heatmap showing influential regions
The heatmap is overlaid on the MRI scan using OpenCV

5. Flask Web Application
Supports image upload
Displays:
Predicted tumor type
Confidence score
Grad-CAM attention map
Additional clinical information (description, severity, recommended action)

Technologies Used
Tools and Libraries
TensorFlow & Keras – model training and fine-tuning
OpenCV & NumPy – preprocessing and Grad-CAM overlay
Flask – deployment and interface
Matplotlib – evaluation visualizations

Coding Standards
PEP8-compliant Python code
Modular functions for prediction, Grad-CAM, and preprocessing
Organized directories (static/, templates/, model, scripts)


Most Challenging Problem Solved
The most challenging issue was achieving high accuracy without overfitting on a small, imbalanced medical dataset. Initially, the model performed inconsistently and failed to generalize. This was addressed by carefully designing a two-phase transfer-learning approach:
Freezing VGG16 layers for stable baseline learning
Unfreezing the final block for targeted fine-tuning at a low learning rate
Combined with augmentation, class balancing, and Grad-CAM visual inspection, the model became significantly more accurate and medically interpretable.
