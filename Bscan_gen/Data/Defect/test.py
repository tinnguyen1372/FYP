from PIL import Image
import numpy as np
import h5py
import os
import math

def preprocess(image_path, res, prefix, angle=0):
    res = int(res)
    img = Image.open(image_path).convert('RGBA')  # Convert the image to RGBA mode

    # Remove the white background and replace with transparency (alpha channel)
    threshold = 240
    img_array = np.array(img)
    img_array[(img_array[:, :, :3] > threshold).all(axis=-1)] = [255, 255, 255, 0]
    img_transparent = Image.fromarray(img_array, 'RGBA')

    # Rotate the image by the specified angle
    img_rotated = img_transparent.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Create a new white image with transparency (alpha channel)
    img_white_background = Image.new("RGBA", img_rotated.size, (255, 255, 255, 255))

    # Paste the rotated content onto the white background
    img_with_background = Image.alpha_composite(img_white_background, img_rotated)

    # Save the rotated and preprocessed image with a "_processed" suffix
    processed_image_path = image_path.replace(".png", "_processed.png")
    img_with_background.save(processed_image_path)

    # Resize the rotated and preprocessed image to the desired resolution
    img_resized = img_with_background.resize((res, res))

    # Convert the resized image to a 2D array of integers
    color_map = {
        (255, 255, 255, 0): -1,  # White (transparent)
        (255, 255, 0, 255): 0,   # Yellow
        (255, 51, 0, 255): 1     # Red
    }

    arr_2d = np.zeros((res, res), dtype=int)
    for y in range(res):
        for x in range(res):
            pixel_color = img_resized.getpixel((x, y))
            arr_2d[y, x] = color_map.get(pixel_color, 0)

    # Expand dimensions
    arr_3d = np.expand_dims(arr_2d, axis=2)

    # Add the angle to the file name
    base_filename = ""
    if "healthy" in image_path:
        base_filename = os.getcwd() + '/' + prefix + 'healthy_angle_{}.h5'.format(angle)
    else:
        base_filename = os.getcwd() + '/' + prefix + 'cavity_angle_{}.h5'.format(angle)

    filename = base_filename
    with h5py.File(filename, 'w') as file:
        dset = file.create_dataset("data", data=arr_3d)
        file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)
        
# Example usage:
image_path = "./defect0.png"
resolution = 200
prefix = "defect0_"
angle = 20  # Set the desired angle for augmentation

preprocess(image_path, resolution, prefix, angle)
