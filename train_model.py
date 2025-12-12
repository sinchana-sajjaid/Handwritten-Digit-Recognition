import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Display dataset information
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")
print(f"Image shape: {x_train[0].shape}")

# Preprocess the data
# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to include channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Preprocessed shape: {x_train[0].shape}")

# Build the CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Display model architecture
model.summary()

# Compile the model
print("\nCompiling model...")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
print("\nTraining model...")
print("This will take a few minutes...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Save the model
print("\nSaving model...")
model.save("digit_recognition_model.h5")
print("Model saved as 'digit_recognition_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")
plt.show()

# Make predictions on sample test images
print("\nMaking predictions on sample images...")
num_samples = 5
predictions = model.predict(x_test[:num_samples])

plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f'Pred: {predicted_label}\nTrue: {true_label}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png')
print("Sample predictions saved as 'sample_predictions.png'")
plt.show()

# Show predictions for ALL digits (0-9)
print("\nShowing examples of ALL digits (0-9)...")
digit_examples = {}

# Find one example of each digit
for i in range(len(y_test)):
    digit = y_test[i]
    if digit not in digit_examples:
        digit_examples[digit] = i
    if len(digit_examples) == 10:
        break

# Make predictions for all digits
all_digit_indices = [digit_examples[i] for i in range(10)]
all_digit_images = x_test[all_digit_indices]
all_predictions = model.predict(all_digit_images)

plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(all_digit_images[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(all_predictions[i])
    true_label = y_test[all_digit_indices[i]]
    confidence = all_predictions[i][predicted_label] * 100
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f'Digit: {true_label}\nPred: {predicted_label}\nConf: {confidence:.1f}%', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Model Predictions for ALL Digits (0-9)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('all_digits_predictions.png')
print("All digits predictions saved as 'all_digits_predictions.png'")
plt.show()

print("\n✅ Project completed successfully!")
print("You can now use the saved model for predictions.")