# Multiclass Classification & Dimensionality Reduction Benchmark

## Overview
Comparative machine learning benchmark evaluating linear, kernel-based, tree-ensemble, and deep neural network architectures on high-dimensional image classification data (70,000 samples, 784 features, 10 classes). Examines the quantitative trade-offs between **PCA dimensionality reduction**, computational runtime, activation functions, and out-of-sample generalization.

## Key Implementations & Findings
- **Dimensionality Reduction (PCA + RBF SVM):** Analyzed singular-value explained variance thresholds (~43 components for 80% variance; ~87 components for 90% variance). Compressing 784 features down to **70 principal components** preserved full-space **93% test accuracy** while cutting SVM training time by **72%**.
- **Neural Network Backpropagation from Scratch (NumPy):** Coded a 3-layer feedforward neural network (`784 -> 128 -> 64 -> 10`) with manual forward/backward propagation and SGD updates. Replacing **Sigmoid** activations with **ReLU** lifted test accuracy from **56% to 91%** over 10 epochs by preventing vanishing gradients.
- **Regularized Deep Neural Network (TensorFlow / Keras):** Built a parameterized DNN incorporating **Dropout regularization (0.2)**, **Adam** optimization, sparse categorical cross-entropy, and **EarlyStopping** callbacks, achieving **98% out-of-sample accuracy**.
- **Baseline & Ensemble Benchmarking:** Benchmarked against L2-regularized **Logistic Regression** (92% accuracy) and **Random Forest** (86% accuracy, 15.8s runtime) with full confusion matrix and per-class F1-score diagnostics.

## Tech Stack
`Python` · `NumPy` · `Scikit-Learn` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn`
