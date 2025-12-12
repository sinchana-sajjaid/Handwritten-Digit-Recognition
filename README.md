# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0-9) using Convolutional Neural Networks (CNN) trained on the MNIST dataset.

## 📋 Overview

This project implements a CNN-based digit recognition system that achieves high accuracy in classifying handwritten digits. The model is trained on the famous MNIST dataset containing 70,000 images of handwritten digits.

## ✨ Features

- **CNN Architecture**: Custom-built convolutional neural network
- **High Accuracy**: Achieves ~98-99% accuracy on test data
- **Complete Pipeline**: Training, evaluation, and prediction scripts
- **Visualization**: Training history plots, confusion matrices, and prediction visualizations
- **Custom Image Prediction**: Can predict digits from your own handwritten images
- **Comprehensive Testing**: Per-digit accuracy analysis and confidence scores

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Metrics and evaluation
- **PIL**: Image processing

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd handwritten-digit-recognition
```

2. **Install required packages**
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
handwritten-digit-recognition/
│
├── train_model.py              # Main training script
├── verify_all_digits.py        # Comprehensive model verification
├── predict_custom.py           # Predict custom images
├── README.md                   # Project documentation
│
├── digit_recognition_model.h5  # Saved trained model
│
└── outputs/                    # Generated visualizations
    ├── training_history.png
    ├── sample_predictions.png
    ├── all_digits_predictions.png
    ├── confusion_matrix_all_digits.png
    ├── all_digits_examples.png
    └── confidence_all_digits.png
```

## 🚀 Usage

### 1. Train the Model

Run the main training script:

```bash
python train_model.py
```

This will:
- Load and preprocess the MNIST dataset
- Build and train the CNN model
- Save the trained model as `digit_recognition_model.h5`
- Generate training history plots
- Show sample predictions for all digits (0-9)

**Expected Output:**
- Training accuracy: ~99%
- Test accuracy: ~98-99%
- Training time: 3-5 minutes (depending on hardware)

### 2. Verify Model Performance

Run comprehensive verification:

```bash
python verify_all_digits.py
```

This script provides:
- Per-digit accuracy (0-9)
- Confusion matrix visualization
- Multiple examples for each digit
- Detailed classification report
- Confidence analysis

### 3. Predict Custom Images

To predict your own handwritten digits:

```bash
python predict_custom.py
```

**Requirements for custom images:**
- Any image format (PNG, JPG, etc.)
- Clear handwritten digit
- Will be automatically resized to 28×28 pixels
- Inverted if necessary (white on black background)

**Steps:**
1. Draw a digit in Paint or any drawing app
2. Save the image (e.g., `my_digit.png`)
3. Update the `image_path` variable in `predict_custom.py`
4. Run the script

## 🧠 Model Architecture

```
Layer (type)                 Output Shape              Parameters
================================================================
Conv2D (32 filters)          (None, 26, 26, 32)        320
MaxPooling2D                 (None, 13, 13, 32)        0
Conv2D (64 filters)          (None, 11, 11, 64)        18,496
MaxPooling2D                 (None, 5, 5, 64)          0
Flatten                      (None, 1600)              0
Dropout (0.5)                (None, 1600)              0
Dense (128 units)            (None, 128)               204,928
Dense (10 units - output)    (None, 10)                1,290
================================================================
Total parameters: 225,034
```

## 📊 Results

### Performance Metrics

- **Overall Test Accuracy**: ~98.5%
- **Training Time**: ~3-5 minutes (5 epochs)
- **Model Size**: ~900 KB

### Per-Digit Performance

| Digit | Accuracy |
|-------|----------|
| 0     | ~99%     |
| 1     | ~99%     |
| 2     | ~98%     |
| 3     | ~98%     |
| 4     | ~98%     |
| 5     | ~98%     |
| 6     | ~99%     |
| 7     | ~98%     |
| 8     | ~97%     |
| 9     | ~97%     |

## 📸 Visualizations

The project generates several visualizations:

1. **Training History**: Accuracy and loss curves over epochs
2. **Sample Predictions**: Example predictions with confidence scores
3. **All Digits Display**: One example of each digit (0-9) with predictions
4. **Confusion Matrix**: Shows model's prediction patterns
5. **Confidence Analysis**: Per-digit confidence visualization
6. **Multiple Examples**: 10 examples of each digit with results

## 🔧 Configuration

You can modify these parameters in `train_model.py`:

```python
# Training parameters
batch_size = 128        # Batch size for training
epochs = 5              # Number of training epochs
validation_split = 0.1  # Validation data percentage

# Model parameters
conv1_filters = 32      # First convolutional layer filters
conv2_filters = 64      # Second convolutional layer filters
dropout_rate = 0.5      # Dropout rate
dense_units = 128       # Dense layer units
```

## 🎯 Use Cases

- **Educational**: Learn CNN architecture and image classification
- **Digit Recognition**: Recognize handwritten digits in forms
- **OCR Systems**: Part of larger OCR (Optical Character Recognition) systems
- **Postal Automation**: Zip code recognition
- **Bank Check Processing**: Amount recognition

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: No file or directory named 'digit_recognition_model.h5'
```
**Solution**: Run `train_model.py` first to create the model.

**2. Low prediction accuracy on custom images**
- Ensure the digit is clearly written
- Use dark digit on light background (or vice versa)
- Center the digit in the image
- Avoid very thick or very thin strokes

**3. Memory errors**
- Reduce `batch_size` in training script
- Close other memory-intensive applications

## 📈 Future Improvements

- [ ] Add data augmentation for improved accuracy
- [ ] Implement real-time webcam digit recognition
- [ ] Create a web interface (Flask/Django)
- [ ] Support for multiple digit sequences
- [ ] Mobile app deployment
- [ ] Transfer learning with other datasets

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- MNIST Dataset: Yann LeCun and Corinna Cortes
- TensorFlow/Keras Team
- Deep Learning Community

---

**Made with ❤️ using TensorFlow and Keras**