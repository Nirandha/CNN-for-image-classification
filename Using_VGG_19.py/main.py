# Import necessary libraries
from keras.applications import VGG19
from keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical

# Load VGG19 with pre-trained weights and exclude the top layers
VGG = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Define the top layers of the model to be added on top of VGG19
x = VGG.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)  # 10 classes in CIFAR-10

# Create the VGG19 model
VGG_model = keras.models.Model(inputs=VGG.input, outputs=x)

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
VGG_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training set
VGG_history = VGG_model.fit(train_images, train_labels_one_hot, epochs=20, batch_size=64, validation_split=0.2)

# Save the trained model to a file
VGG_model.save('VGG19.h5')
print("Saved model to disk")

# Save the training history to a CSV file
df = pd.DataFrame(VGG_history.history)
df.to_csv("VGG_history.csv")

# Load the trained model
loaded_VGG_model = keras.models.load_model('VGG19.h5', custom_objects={"accuracy": "accuracy"})

# Evaluate the model on the test set
evaluation = loaded_VGG_model.evaluate(test_images, test_labels_one_hot, verbose=1)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')

# Display the model summary
loaded_VGG_model.summary()

# Predict on test set and create a confusion matrix
y_pred = loaded_VGG_model.predict(test_images)
cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Read and plot the training loss from the CSV file
df = pd.read_csv('VGG_history.csv')
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
df = pd.read_csv('VGG_history.csv')
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
