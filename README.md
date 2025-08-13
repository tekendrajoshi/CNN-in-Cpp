# 🧠 CNN from Scratch in C++

This repository contains a **Convolutional Neural Network (CNN)** implementation **from scratch** in **C++**, without using machine learning libraries like TensorFlow or PyTorch.  
It demonstrates the core principles of convolution, activation functions, pooling, flattening, and fully connected layers — all built manually.

## 📌 Features
- Load and parse MNIST dataset (`mnist_train.csv`)
- Random filter initialization for convolution layers
- Forward pass:
  - Convolution
  - ReLU activation
  - Max pooling
  - Flattening
  - Fully connected layer
- Cross-entropy loss calculation
- Simple backpropagation for fully connected layer
- Multiple training epochs
- Accuracy and loss tracking per epoch

## 📂 Project Structure
- `main.cpp` — main CNN training loop
- `functions.h` — helper functions for convolution, pooling, etc.
- `mnist_train.csv` — dataset file (download separately)


