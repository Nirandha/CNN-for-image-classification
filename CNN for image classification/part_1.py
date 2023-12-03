# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, BatchNormalization, Activation, Conv2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow import keras
import cv2
import torchvision
import torchvision.transforms as transforms
from keras.datasets import cifar10

# Define data transformations for PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training dataset using torchvision
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# Load the testing dataset using torchvision
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Load the CIFAR-10 dataset using Keras
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the shapes of the training images and labels
print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)

# Convert categorical labels to one-hot encoding for training set
train_labels_one_hot = to_categorical(train_labels, num_classes=10)

# Convert categorical labels to one-hot encoding for testing set
test_labels_one_hot = to_categorical(test_labels, num_classes=10)

# Define the model architecture
def build_model(inputs, filters, dropout_rate, kernel_size=3): 
    """
    Build a convolutional neural network model.

    Parameters:
    - inputs: Input tensor to the model.
    - filters: List of three integers representing the number of filters for each convolutional layer.
    - dropout_rate: Dropout rate for regularization.
    - kernel_size: Size of the convolutional kernel.

    Returns:
    - Output tensor of the model.
    """

    # First Convolutional Layer
    x = Conv2D(filters[0], kernel_size, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max Pooling Layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Convolutional Layer
    x = Conv2D(filters[1], kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max Pooling Layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output for the fully connected layers
    x = Flatten()(x)

    # Fully Connected Layer
    x = Dense(filters[2], activation='relu')(x)
    
    # Dropout for regularization
    x = Dropout(dropout_rate)(x)

    # Output Layer with 10 classes and softmax activation
    x = Dense(10, activation='softmax')(x)

    return x
