import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Function to load and preprocess images
def load_data(image_path, target_size=(256, 256)):
    image_files = glob(os.path.join(image_path, "*.png"))
    images = []
    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img = cv2.resize(img, target_size)
        images.append(img)
    return np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]

# Custom loss function to encourage better colorization and noise removal
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# U-Net-like architecture for colorization and denoising
def build_colorizer(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)
    
    # Encoder: Downsampling
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x1)  # Downsample
    
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x2)  # Downsample
    
    # Decoder: Upsampling
    x3 = UpSampling2D((2, 2))(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)
    
    x4 = UpSampling2D((2, 2))(x3)
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
    
    # Output colorized layer (3 channels for RGB output)
    outputs = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(x4)
    
    # Residual connection for denoising
    denoised = Add()([outputs, Conv2D(3, (1, 1), activation='linear', padding='same')(x4)])
    
    model = Model(inputs, denoised)
    return model

def main():
    # Load dataset
    image_path = "path_to_your_sar_images"
    images = load_data(image_path)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    # Split into training and validation sets
    X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)

    # Build the model
    model = build_colorizer(input_shape=(256, 256, 1))
    model.compile(optimizer='adam', loss=custom_loss)

    # Set up callbacks
    checkpoint = ModelCheckpoint("colorizer_model.h5", monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=50, batch_size=8, callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    main()
