import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def feature_extractor(inputs):
    # Use ResNet50 as a feature extractor
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    # Global Average Pooling and dense layers for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    # Upsample the input, use ResNet as feature extractor, and add classifier
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

def define_compile_model():
    # Define, compile, and return the model
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Assuming 'train_images', 'train_labels', 'test_images', 'test_labels' are loaded previously

# Create and train the ResNet model
resnet_model = define_compile_model()
resnet_history = resnet_model.fit(train_images, train_labels, validation_split=0.2, epochs=5, batch_size=64)

# Save the trained model and training history to files
resnet_model.save('resnet_model.h5')
df = pd.DataFrame(resnet_history.history)
df.to_csv("resnet_history.csv")

# Load the trained model
resnet_model = tf.keras.models.load_model('resnet_model.h5', custom_objects={"accuracy": "accuracy"})

# Evaluate the model on the test set
evaluation = resnet_model.evaluate(test_images, test_labels, verbose=1)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')

# Predict on test set and create a confusion matrix
y_pred = resnet_model.predict(test_images)
cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Read and plot the training loss from the CSV file
df = pd.read_csv('resnet_history.csv')
loss = df['loss']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(loss) + 1)
ax.plot(x_values, loss, label='Loss')
ax.set_xticks(np.arange(1, 6))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss over Epochs')
ax.legend()
plt.show()

# Read and plot the training accuracy from the CSV file
df = pd.read_csv('resnet_history.csv')
accuracy = df['accuracy']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(accuracy) + 1)
ax.plot(x_values, accuracy, label='Accuracy')
ax.set_xticks(np.arange(1, 6))
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy over Epochs')
ax.legend()
plt.show()
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def feature_extractor(inputs):
    # Use ResNet50 as a feature extractor
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    # Global Average Pooling and dense layers for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    # Upsample the input, use ResNet as feature extractor, and add classifier
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

def define_compile_model():
    # Define, compile, and return the model
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Assuming 'train_images', 'train_labels', 'test_images', 'test_labels' are loaded previously

# Create and train the ResNet model
resnet_model = define_compile_model()
resnet_history = resnet_model.fit(train_images, train_labels, validation_split=0.2, epochs=5, batch_size=64)

# Save the trained model and training history to files
resnet_model.save('resnet_model.h5')
df = pd.DataFrame(resnet_history.history)
df.to_csv("resnet_history.csv")

# Load the trained model
resnet_model = tf.keras.models.load_model('resnet_model.h5', custom_objects={"accuracy": "accuracy"})

# Evaluate the model on the test set
evaluation = resnet_model.evaluate(test_images, test_labels, verbose=1)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')

# Predict on test set and create a confusion matrix
y_pred = resnet_model.predict(test_images)
cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))

# Plot the confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Read and plot the training loss from the CSV file
df = pd.read_csv('resnet_history.csv')
loss = df['loss']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(loss) + 1)
ax.plot(x_values, loss, label='Loss')
ax.set_xticks(np.arange(1, 6))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss over Epochs')
ax.legend()
plt.show()

# Read and plot the training accuracy from the CSV file
df = pd.read_csv('resnet_history.csv')
accuracy = df['accuracy']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(accuracy) + 1)
ax.plot(x_values, accuracy, label='Accuracy')
ax.set_xticks(np.arange(1, 6))
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy over Epochs')
ax.legend()
plt.show()
