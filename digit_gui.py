import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

MODEL_PATH = "digit_recognition_model.h5"  # your trained model file
CANVAS_SIZE = 280   # drawing area size (pixels)
IMG_SIZE = 28       # model input size (28x28)
BG_COLOR = "black"  # background color
FG_COLOR = "white"  # drawing color (digit)
BRUSH_SIZE = 15     # thickness of the stroke


class DigitApp:
    def __init__(self, model):
        self.model = model

        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition (CNN)")

        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BG_COLOR
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # PIL image that mirrors the canvas (for prediction)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black background
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)

        # Buttons
        predict_btn = tk.Button(self.root, text="Predict", command=self.predict_digit)
        predict_btn.grid(row=1, column=0, pady=10, sticky="ew")

        clear_btn = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        clear_btn.grid(row=1, column=1, pady=10, sticky="ew")

        exit_btn = tk.Button(self.root, text="Exit", command=self.root.destroy)
        exit_btn.grid(row=1, column=2, pady=10, sticky="ew")

        # Label to show result
        self.result_label = tk.Label(self.root, text="Draw a digit (0–9) and click Predict", font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=10)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_stroke(self, event):
        x, y = event.x, event.y
        # Draw on Tkinter canvas
        self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            fill=FG_COLOR,
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )
        # Draw on PIL image (for model input)
        self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=BRUSH_SIZE)
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="Draw a digit (0–9) and click Predict")

    def preprocess_image(self):
        # Resize to 28x28 as MNIST
        img = self.image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img_array = np.array(img)

        # MNIST digits are white on black; we already drew white on black,
        # but just in case, ensure correct range and shape.
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        return img_array

    def predict_digit(self):
        # Prepare the image
        img_array = self.preprocess_image()

        # Run prediction
        preds = self.model.predict(img_array)
        digit = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)

        self.result_label.config(
            text=f"Predicted: {digit}   |   Confidence: {confidence:.2f}%"
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model file:\n{e}")
        raise

    app = DigitApp(model)
    app.run()