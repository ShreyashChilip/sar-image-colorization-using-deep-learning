import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configurations
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SAR_CHANNELS = 1  # Grayscale input for SAR

# Load the trained model
model = load_model('sar_colorization_model.h5')

# Preprocess SAR image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255.0
    return image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

# Post-process and save the colorized image
def save_colorized_image(image, output_path):
    image = np.clip(image[0], 0, 1) * 255.0  # Convert back to [0, 255]
    image = image.astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved at: {output_path}")

# Main function to colorize a given SAR image
def colorize_image(input_image_path, output_image_path):
    input_image = preprocess_image(input_image_path)
    predicted_color = model.predict(input_image)
    save_colorized_image(predicted_color, output_image_path)

if __name__ == '__main__':
    # Input and output paths for the SAR image and the colorized output
    input_sar_image = 'path_to_input_sar_image/sar_image.png'  # Specify the path to the input SAR image
    output_color_image = 'path_to_output_image/colorized_output.png'  # Specify the output path

    colorize_image(input_sar_image, output_color_image)
