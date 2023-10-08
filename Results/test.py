import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import log10

image1 = cv2.imread('cavity-0.png')
image2 = cv2.imread('cavity-0_src.png')
# Ensure that both images have the same dimensions
if image1.shape != image2.shape:
    # Resize one of the images to match the dimensions of the other
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
print()

# Convert the images to grayscale (required for SSIM calculation)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


def calculate_mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse
def calculate_mae(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(np.abs(diff))
    mae = err/(float(h*w))
    return mae
# Calculate SSIM
ssim_value = ssim(gray_image1, gray_image2)

# Calculate MAE (Mean Absolute Error)
mae = calculate_mae(gray_image1,gray_image2)
# Calculate MSE using your custom function
mse = calculate_mse(gray_image1, gray_image2)

# Calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(image1, image2):
    non_zero_indices = (image1 != 0)
    diff = np.abs(image1.astype(float) - image2.astype(float))
    mape = np.mean(np.divide(diff, image1.astype(float), out=np.zeros_like(diff), where=non_zero_indices)) * 100
    return mape

mape_value = calculate_mape(gray_image1, gray_image2)

# Calculate MRE (Mean Relative Error)
def calculate_mre(image1, image2):
    absolute_image1 = np.abs(image1.astype(float))
    absolute_image2 = np.abs(image2.astype(float))
    non_zero_indices = (absolute_image1 != 0) | (absolute_image2 != 0)
    relative_diff = np.abs(image1.astype(float) - image2.astype(float)) / np.maximum(absolute_image1, absolute_image2)
    mre = np.mean(np.divide(relative_diff, np.maximum(absolute_image1, absolute_image2), out=np.zeros_like(relative_diff), where=non_zero_indices)) * 100
    return mre

mre_value = calculate_mre(gray_image1, gray_image2)

# Print the calculated values
print(f"SSIM: {ssim_value}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"MAPE: {mape_value}%")
print(f"MRE: {mre_value}%")
