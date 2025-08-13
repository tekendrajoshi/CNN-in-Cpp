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

## 📊 Example Output
- Skipping row: stoi
- Loaded 60000 images (1 skipped) from mnist_train.csv
- Total images loaded: 60000
- Total labels loaded: 60000
- Starting forward pass for all images...
- Epoch 1: Forward pass for all images...
- Epoch 1 - Accuracy: 85.3967%
- Epoch 1 - Average Loss: 0.654386
- Epoch 2: Forward pass for all images...
- Epoch 2 - Accuracy: 89.5633%
- Epoch 2 - Average Loss: 0.389061
- Epoch 3: Forward pass for all images...
- Epoch 3 - Accuracy: 90.335%
- Epoch 3 - Average Loss: 0.34657
- Epoch 4: Forward pass for all images...
- Epoch 4 - Accuracy: 90.8467%
- Epoch 4 - Average Loss: 0.324442
- Epoch 5: Forward pass for all images...
- Epoch 5 - Accuracy: 91.205%
- Epoch 5 - Average Loss: 0.309795
- Epoch 6: Forward pass for all images...
- Epoch 6 - Accuracy: 91.51%
- Epoch 6 - Average Loss: 0.298925



