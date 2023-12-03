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

# Assume 'build_model' function is defined here

# Load the CIFAR-10 dataset using Keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

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
    # ... (unchanged code)

# Define input shape for the model
input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)

# Build the model with specified parameters
output = build_model(inputs, filters=[64, 128, 256], dropout_rate=0.01)

# Define a custom optimizer with a higher learning rate for demonstration
custom_optimizer = keras.optimizers.Adam(learning_rate=0.1)

# Create a Keras Model with the specified input and output
model_1 = keras.Model(inputs=inputs, outputs=output)

# Compile the model with categorical crossentropy loss, custom optimizer, and accuracy metric
model_1.compile(loss='categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

# Train the model on the training data
history = model_1.fit(train_images, train_labels_one_hot, batch_size=64, epochs=20, verbose="auto", validation_split=0.2)

# Save the trained model to a file
model_1.save('cifar10_1.h5')
print("Saved model to disk")

# Load the saved model
loaded_model = keras.models.load_model('cifar10_1.h5')

# Evaluate the model on the test data
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_labels_one_hot)
print(f'Test accuracy: {test_accuracy}')

# Assuming you have a trained model and test data X_test
y_pred = loaded_model.predict(test_images)

# Create a confusion matrix
cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Read and plot the training history from the CSV file
df = pd.read_csv('history_1.csv')

# Plot the training loss
loss = df['loss']
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(loss, label='Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss over Epochs')
ax.legend()
plt.show()

# Plot the validation loss
val_loss = df['val_loss']
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(val_loss, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss over Epochs')
ax.legend()
plt.show()

# Plot the training accuracy
accuracy = df['accuracy']
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(accuracy, label='Training Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy over Epochs')
ax.legend()
plt.show()
