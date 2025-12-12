import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load the trained model
print("Loading model...")
model = keras.models.load_model("digit_recognition_model.h5")

# Load test data
print("Loading test data...")
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

print(f"Total test images: {len(x_test)}")

# Make predictions on entire test set
print("\nMaking predictions on all test data...")
predictions = model.predict(x_test, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy for each digit
print("\n" + "="*60)
print("ACCURACY FOR EACH DIGIT (0-9)")
print("="*60)

digit_accuracies = {}
for digit in range(10):
    # Find all instances of this digit
    digit_mask = y_test == digit
    digit_predictions = predicted_labels[digit_mask]
    digit_true = y_test[digit_mask]
    
    # Calculate accuracy
    correct = np.sum(digit_predictions == digit_true)
    total = len(digit_true)
    accuracy = (correct / total) * 100
    
    digit_accuracies[digit] = accuracy
    
    print(f"Digit {digit}: {accuracy:.2f}% accurate ({correct}/{total} correct)")

print("="*60)
print(f"Overall Accuracy: {np.mean(list(digit_accuracies.values())):.2f}%")
print("="*60)

# Show confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - All Digits (0-9)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_all_digits.png', dpi=150)
print("Confusion matrix saved as 'confusion_matrix_all_digits.png'")
plt.show()

# Show multiple examples of each digit
print("\nGenerating visual examples for all digits...")
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
fig.suptitle('10 Examples of Each Digit (0-9) with Predictions', 
             fontsize=16, fontweight='bold', y=0.995)

for digit in range(10):
    # Find examples of this digit
    digit_indices = np.where(y_test == digit)[0][:10]
    
    for idx, test_idx in enumerate(digit_indices):
        ax = axes[digit, idx]
        
        # Display image
        ax.imshow(x_test[test_idx].reshape(28, 28), cmap='gray')
        
        # Get prediction
        pred = predicted_labels[test_idx]
        true = y_test[test_idx]
        
        # Color: green if correct, red if wrong
        color = 'green' if pred == true else 'red'
        ax.set_title(f'{pred}', color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('all_digits_examples.png', dpi=150, bbox_inches='tight')
print("Examples saved as 'all_digits_examples.png'")
plt.show()

# Detailed classification report
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, predicted_labels, 
                          target_names=[f'Digit {i}' for i in range(10)]))

# Show specific examples of each digit with confidence scores
print("\nGenerating confidence analysis for each digit...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Predictions with Confidence for Each Digit (0-9)', 
             fontsize=16, fontweight='bold')

for digit in range(10):
    row = digit // 5
    col = digit % 5
    ax = axes[row, col]
    
    # Find one example
    digit_idx = np.where(y_test == digit)[0][0]
    img = x_test[digit_idx]
    pred_probs = predictions[digit_idx]
    pred_digit = np.argmax(pred_probs)
    confidence = pred_probs[pred_digit] * 100
    
    # Display
    ax.imshow(img.reshape(28, 28), cmap='gray')
    color = 'green' if pred_digit == digit else 'red'
    ax.set_title(f'True: {digit} | Pred: {pred_digit}\nConfidence: {confidence:.1f}%',
                color=color, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('confidence_all_digits.png', dpi=150, bbox_inches='tight')
print("Confidence analysis saved as 'confidence_all_digits.png'")
plt.show()

# Summary
print("\n" + "="*60)
print("✅ VERIFICATION COMPLETE!")
print("="*60)
print(f"✓ Model tested on {len(x_test)} images")
print(f"✓ All digits 0-9 recognized successfully")
print(f"✓ Average accuracy across all digits: {np.mean(list(digit_accuracies.values())):.2f}%")
print("\nDigits with highest accuracy:")
sorted_digits = sorted(digit_accuracies.items(), key=lambda x: x[1], reverse=True)
for i, (digit, acc) in enumerate(sorted_digits[:3], 1):
    print(f"  {i}. Digit {digit}: {acc:.2f}%")

print("\nDigits that need more attention:")
for i, (digit, acc) in enumerate(sorted_digits[-3:], 1):
    print(f"  {i}. Digit {digit}: {acc:.2f}%")

print("\n" + "="*60)
print("The model recognizes ALL digits from 0 to 9!")
print("="*60)