import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras import layers, models

import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras

def load_and_preprocess_input_image(image_path, target_size=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Grayscale image with a single channel
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 in the range [0, 1]
    image = tf.image.resize(image, target_size)
    return image

input_images = []

# Loop through the image files
for i in range(1, 11):  # Assuming images from 1 to 10 are present
    input_healthy_image_name = f'healthy-{i}.png'
    input_cavity_image_name = f'cavity-{i}.png'

    input_healthy_image_path = os.path.join(input_healthy_image_name)
    input_cavity_image_path = os.path.join(input_cavity_image_name)

    if os.path.exists(input_healthy_image_path):
        input_image = load_and_preprocess_input_image(input_healthy_image_path)
        input_images.append(input_image)

    if os.path.exists(input_cavity_image_path):
        input_image = load_and_preprocess_input_image(input_cavity_image_path)
        input_images.append(input_image)
image_size = 128

def multi_scale(x, filters, padding="same", strides=1):
    c1 = keras.layers.Conv2D(filters, (1,1), padding=padding, strides=strides, activation="relu")(x)
    c3 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c5 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c5 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c5)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(x)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c7)
    c7 = keras.layers.Conv2D(filters, (3,3), padding=padding, strides=strides, activation="relu")(c7)
    c = keras.layers.Concatenate()([c1, c3, c5, c7])
    return c

def down_block2(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    c = multi_scale(x, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block2(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.Conv2D(filters, (2, 2), padding='same', strides=1, activation="relu")(us)
    c = keras.layers.Concatenate()([us, skip])
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck2(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    f = int(filters/4)
    c = multi_scale(x, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = multi_scale(c, f)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

# Convert the input_images list to a NumPy array
input_images = np.array(input_images)

# Define the path to the pre-trained model
model_path = 'unet_trained_model-Aug-5.h5'


# Define the U-Net model
def unet(input_shape, num_classes):
    
    f0 = 16
    f = [f0, f0*2, f0*4, f0*8, f0*16]
    
    inputs = tf.keras.Input(shape=input_shape)

     #### model_1
    p0 = inputs
    c1, p1 = down_block2(p0, f[0])
    c2, p2 = down_block2(p1, f[1])
    c3, p3 = down_block2(p2, f[2])
    c4, p4 = down_block2(p3, f[3])
    
    bn = bottleneck2(p4, f[4])
    
    u1 = up_block2(bn, c4, f[3])
    u2 = up_block2(u1, c3, f[2])
    u3 = up_block2(u2, c2, f[1])
    u4 = up_block2(u3, c1, f[0])
    
    output_1 = keras.layers.Conv2D(1, (1, 1), padding="same", activation="relu")(u4)
    model = tf.keras.Model(inputs, output_1)
    
    return model

# Example usage
input_shape = (128, 128, 1)  # Define the shape of your input images (e.g., 256x256 RGB images)
num_classes = 1  # Define the number of channels in the output image (e.g., 3 for RGB)

# Create the U-Net model
model = unet(input_shape, num_classes)

model.load_weights(model_path)
# Apply the model to the input images
predicted_output_data = model.predict(input_images)

# Save the predicted output images
for i, pred in enumerate(predicted_output_data):
    pred = (pred.reshape(128, 128) * 255).astype(np.uint8)  # Assuming the output shape is (128, 128, 1)
    pred = Image.fromarray(pred, mode='L')
    pred.save(f'output_image_{i + 1}.png')
