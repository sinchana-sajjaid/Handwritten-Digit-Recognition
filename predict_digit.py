import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path):
    """
    Preprocess an image for prediction
    """
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert if necessary (MNIST digits are white on black)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array, img

def predict_digit(model_path, image_path):
    """
    Predict the digit in an image
    """
    # Load the trained model
    print("Loading model...")
    model = keras.models.load_model(model_path)
    
    # Preprocess the image
    print(f"Processing image: {image_path}")
    processed_img, original_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_img, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    
    # Display results
    plt.figure(figsize=(10, 4))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Predicted: {predicted_digit} (Confidence: {confidence:.2f}%)')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    print(f"\n✅ Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show all probabilities
    print("\nAll probabilities:")
    for i in range(10):
        print(f"  Digit {i}: {prediction[0][i]*100:.2f}%")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    model_path = "digit_recognition_model.h5"
    image_path = "test_digit.png"  # Change this to your image file
    
    try:
        predict_digit(model_path, image_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nInstructions:")
        print("1. Make sure you've trained the model first (run the main script)")
        print("2. Create or download a handwritten digit image")
        print("3. Update 'image_path' variable with your image file name")
        print("\nTip: Draw a digit in Paint/similar app, save as PNG, and place it in the same folder")