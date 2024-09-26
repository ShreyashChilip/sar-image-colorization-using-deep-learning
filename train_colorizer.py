import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Configurations
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SAR_CHANNELS = 1  # SAR images are grayscale
COLOR_CHANNELS = 3  # RGB output channels

# Define U-Net Model Architecture
def build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, SAR_CHANNELS)):
    inputs = Input(shape=input_shape)

    # Encoder (Downsampling)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)

    # Decoder (Upsampling)
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    merge1 = Concatenate()([up1, conv2])

    up2 = UpSampling2D(size=(2, 2))(merge1)
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    merge2 = Concatenate()([up2, conv1])

    # Output Layer for Colorization
    output = Conv2D(COLOR_CHANNELS, (1, 1), activation='sigmoid')(merge2)

    return Model(inputs, output)

# Data Preparation
def load_image_pairs(sar_image_dir, color_image_dir):
    sar_images = []
    color_images = []
    
    # List of all SAR images in the folder
    for filename in os.listdir(sar_image_dir):
        if filename.endswith('.png'):
            # Load SAR image
            sar_path = os.path.join(sar_image_dir, filename)
            sar_image = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
            sar_image = cv2.resize(sar_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            sar_image = sar_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Load corresponding color image
            color_path = os.path.join(color_image_dir, filename)
            if os.path.exists(color_path):
                color_image = cv2.imread(color_path)
                color_image = cv2.resize(color_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                color_image = color_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

                # Add images to lists
                sar_images.append(sar_image[..., np.newaxis])  # Add channel dimension for SAR
                color_images.append(color_image)

    # Convert lists to numpy arrays
    return np.array(sar_images), np.array(color_images)

def main():
    # Directory paths
    sar_image_dir = 'path_to_sar_images'  # e.g., './data/sar_images'
    color_image_dir = 'path_to_color_images'  # e.g., './data/color_images'
    
    # Load paired images
    X, y = load_image_pairs(sar_image_dir, color_image_dir)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Model
    model = build_unet()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test))

    # Save Model
    model.save('sar_colorization_model.h5')
    print("Model saved as 'sar_colorization_model.h5'")

if __name__ == '__main__':
    main()
