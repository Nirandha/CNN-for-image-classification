import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



from tensorflow import keras
from keras.layers import MaxPooling2D, BatchNormalization, Activation, Conv2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.layers import Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.layers import GlobalAveragePooling2D
import cv2


# for pytorch
import torchvision
import torchvision.transforms as transforms

input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)
output = build_model(inputs, filters=[64, 128, 256], d=0.01)
custom_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model_0001 = keras.Model(inputs=inputs, outputs=output)
model_0001.compile(loss='categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])




history = model_0001.fit(train_images, train_labels_one_hot, batch_size=64, epochs=20, verbose="auto", validation_split=0.2)
model_0001.save('cifar10_0001.h5')
print("Saved model to disk")

df = pd.DataFrame(history.history)
df.to_csv("history_0001.csv")

model = keras.models.load_model('cifar10_0001.h5')


test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)
print(f'Test accuracy: {test_accuracy}')

# Assuming you have a trained model and test data X_test
y_pred = model.predict(test_images)

cm = confusion_matrix(test_labels, np.argmax(y_pred, axis=1))
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# reading the csv file
df = pd.read_csv('history_0001.csv')
loss = df['loss']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(loss) + 1)
ax.plot(x_values, loss, label='loss')
ax.set_xticks(np.arange(1, 21))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Training Loss')
ax.legend()
plt.show()


# reading the csv file
df = pd.read_csv('history_0001.csv')
loss = df['val_loss']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(loss) + 1)
ax.plot(x_values, loss, label='loss')
ax.set_xticks(np.arange(1, 21))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Validation Loss')
ax.legend()
plt.show()

# reading the csv file
df = pd.read_csv('history_0001.csv')
accuracy = df['accuracy']
fig, ax = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(accuracy) + 1)
ax.plot(x_values, accuracy, label='accuracy')
ax.set_xticks(np.arange(1, 21))
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy')
ax.legend()
plt.show()