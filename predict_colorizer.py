import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Function to load a single image for prediction
def load_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img / 255.0  # Normalize

# Function to save the colorized output
def save_output(output, save_path):
    output = (output * 255).astype(np.uint8)
    cv2.imwrite(save_path, output)

# Function to perform prediction
def predict_colorize(image_path, model_path, save_path):
    # Load model
    model = load_model(model_path, compile=False)

    # Load image
    img = load_image(image_path)
    
    # Predict and colorize
    prediction = model.predict(img)[0]
    
    # Save the output image
    save_output(prediction, save_path)
    print(f"Colorized image saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    input_image = "path_to_your_input_sar_image.png"  # Grayscale SAR image path
    output_image = "path_to_save_colorized_image.png"  # Output path
    model_file = "colorizer_model.h5"  # Trained model file

    # Perform colorization
    predict_colorize(input_image, model_file, output_image)
