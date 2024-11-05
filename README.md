# Import Libraries

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

# Load and Prepare CIFAR-10 Dataset

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

# Class Names for CIFAR-10 Categories

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize the First 10 Images in Training Data

plt.figure(figsize=(10,10))

for i in range(10):

 plt.subplot(5, 5, i + 1)

 plt.xticks([])

 plt.yticks([])

 plt.grid(False)

 plt.imshow(train_images[i])

 plt.xlabel(class_names[train_labels[i][0]])

plt.show()

# Build the CNN Model

model = models.Sequential([

 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

 layers.MaxPooling2D((2, 2)),

 layers.Conv2D(64, (3, 3), activation='relu'),

 layers.MaxPooling2D((2, 2)),

 layers.Conv2D(64, (3, 3), activation='relu'),

 layers.Flatten(),

 layers.Dense(64, activation='relu'),

 layers.Dense(32, activation='relu'),

 layers.Dense(10)

])

# Model Summary

model.summary()

# Compile and Train the Model

model.compile(optimizer='adam',

 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

 metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 

 validation_data=(test_images, test_labels))

# Plot Training and Validation Accuracy/Loss

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Accuracy/Loss')

plt.ylim([0, 1])

plt.legend(loc='lower right')

plt.show()

# Plot Training and Validation Accuracy

plt.figure(figsize=(8, 4))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')


plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0, 1])

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')

plt.show()

# Plot Training and Validation Loss

plt.figure(figsize=(8, 4))

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.ylim([0, 1.5])

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()

# Combined Plot for Accuracy and Loss

fig, ax1 = plt.subplots(figsize=(8, 4))

# Plot accuracy on the left y-axis

ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')

ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='cyan')

ax1.set_xlabel('Epoch')

ax1.set_ylabel('Accuracy')

ax1.set_ylim([0, 1])

ax1.legend(loc='upper left')

# Create a second y-axis for the loss

ax2 = ax1.twinx()

ax2.plot(history.history['loss'], label='Training Loss', color='red')

ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')

ax2.set_ylabel('Loss')

ax2.set_ylim([0, 1.5])

ax2.legend(loc='upper right')

plt.title('Training and Validation Accuracy and Loss')

plt.show()
