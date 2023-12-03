# Import necessary libraries
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical

# Load the InceptionV3 model with ImageNet weights, excluding the top (fully connected) layers
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

# Resize the original images to fit the input shape of InceptionV3
resized_train_images = [cv2.resize(image, (75, 75)) for image in train_images]
resized_test_images = [cv2.resize(image, (75, 75)) for image in test_images]

resized_train_images = np.array(resized_train_images)
resized_test_images = np.array(resized_test_images)

# Display the shapes of the original and resized images
train_images.shape, resized_train_images.shape, resized_test_images.shape

# Define the top layers of the model to be added on top of InceptionV3
x = inception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='relu')(x)

# Configure the optimizer with a learning rate and clip value
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

# Create a new model combining InceptionV3 base and top layers
inception_model_with_top = keras.models.Model(inputs=inception_model.input, outputs=x)

# Compile the model with categorical crossentropy loss, the configured optimizer, and accuracy metric
inception_model_with_top.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model on the resized training images
inception_history = inception_model_with_top.fit(resized_train_images, train_labels_one_hot, batch_size=64, epochs=20, verbose="auto", validation_split=0.2)

# Save the trained model to a file
inception_model_with_top.save('inception_model.h5')
print("Saved model to disk")

# Save the training history to a CSV file
df = pd.DataFrame(inception_history.history)
df.to_csv("inception_history.csv")

# Load the trained model
loaded_inception_model = keras.models.load_model('inception_model.h5')

# Display the model summary
loaded_inception_model.summary()

# Evaluate the model on the resized test images
evaluation = loaded_inception_model.evaluate(resized_test_images, test_labels_one_hot, verbose=1)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')

# Predict on test images and create a confusion matrix
y_pred = loaded_inception_model.predict(resized_test_images)
cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Read and plot the training loss from the CSV file
df = pd.read_csv('inception_history.csv')
loss = df['loss']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(loss) + 1)
ax.plot(x_values, loss, label='Loss')
ax.set_xticks(np.arange(1, 21))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss over Epochs')
ax.legend()
plt.show()

# Read and plot the training accuracy from the CSV file
df = pd.read_csv('inception_history.csv')
accuracy = df['accuracy']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(accuracy) + 1)
ax.plot(x_values, accuracy, label='Accuracy')
ax.set_xticks(np.arange(1, 21))
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy over Epochs')
ax.legend()
plt.show()
