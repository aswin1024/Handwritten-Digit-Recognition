# Handwritten Digit Recognition using CNN (with local MNIST dataset)
# ----------------------------------------------------------
# Author: Ashwindev Anoop 
# Libraries: TensorFlow / Keras / NumPy / Matplotlib
# ----------------------------------------------------------

import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Step 1: Load MNIST Data from local idx files
# ----------------------------------------------------------
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return data

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.uint8)
    return data

# Change the file paths if needed (same folder as this script)
X_train = load_images('train-images.idx3-ubyte')
y_train = load_labels('train-labels.idx1-ubyte')
X_test = load_images('t10k-images.idx3-ubyte')
y_test = load_labels('t10k-labels.idx1-ubyte')

print("✅ Data loaded successfully!")
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# ----------------------------------------------------------
# Step 2: Data Preprocessing
# ----------------------------------------------------------
# Normalize pixel values (0–1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN input (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ----------------------------------------------------------
# Step 3: Build CNN Model
# ----------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------------------------------------
# Step 4: Train the Model
# ----------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ----------------------------------------------------------
# Step 5: Evaluate Model
# ----------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# ----------------------------------------------------------
# Step 6: Visualize Predictions
# ----------------------------------------------------------
predictions = model.predict(X_test[:9])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:9], axis=1)

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {predicted_classes[i]} | True: {true_classes[i]}")
    plt.axis('off')
plt.show()

# ----------------------------------------------------------
# Step 7: Save the model (optional)
# ----------------------------------------------------------
model.save("handwritten_digit_model.h5")
print("💾 Model saved as handwritten_digit_model.h5")
