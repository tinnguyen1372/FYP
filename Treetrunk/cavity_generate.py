import h5py
import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath


# Set the resolution of the binary array
resolution = 30  # Adjust as needed for desired resolution
import math
import random
def generate_random_convex_polygon(n):
    x_pool = [random.random() for _ in range(n)]
    y_pool = [random.random() for _ in range(n)]

    x_pool.sort()
    y_pool.sort()

    min_x = x_pool[0]
    max_x = x_pool[n - 1]
    min_y = y_pool[0]
    max_y = y_pool[n - 1]

    x_vec = []
    y_vec = []
    last_top = min_x
    last_bot = min_x

    for i in range(1, n - 1):
        x = x_pool[i]

        if random.choice([True, False]):
            x_vec.append(x - last_top)
            last_top = x
        else:
            x_vec.append(last_bot - x)
            last_bot = x

    x_vec.append(max_x - last_top)
    x_vec.append(last_bot - max_x)

    last_left = min_y
    last_right = min_y

    for i in range(1, n - 1):
        y = y_pool[i]

        if random.choice([True, False]):
            y_vec.append(y - last_left)
            last_left = y
        else:
            y_vec.append(last_right - y)
            last_right = y

    y_vec.append(max_y - last_left)
    y_vec.append(last_right - max_y)

    random.shuffle(y_vec)

    vec = [Point2D(x, y) for x, y in zip(x_vec, y_vec)]

    vec.sort(key=lambda v: math.atan2(v.y, v.x))

    x = 0
    y = 0
    min_polygon_x = 0
    min_polygon_y = 0
    points = []

    for i in range(n):
        points.append(Point2D(x, y))

        x += vec[i].x
        y += vec[i].y

        min_polygon_x = min(min_polygon_x, x)
        min_polygon_y = min(min_polygon_y, y)

    x_shift = min_x - min_polygon_x
    y_shift = min_y - min_polygon_y

    for i in range(n):
        p = points[i]
        points[i] = Point2D(p.x + x_shift, p.y + y_shift)

    return points

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Generate the convex polygon points
n = 40  # Number of points
points = generate_random_convex_polygon(n)

# Extract x and y coordinates
x_coords = [point.x for point in points]
y_coords = [point.y for point in points]

# Create a matplotlib Path object from the polygon points
polygon_path = mpath.Path(np.column_stack((x_coords, y_coords)))

# Define the bounding box to encompass the polygon
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)


# Generate the binary array representing the polygon
x_vals = np.linspace(x_min, x_max, resolution)
y_vals = np.linspace(y_min, y_max, resolution)

outline_array = np.zeros((resolution, resolution), dtype=int)

for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        if polygon_path.contains_point((x, y)):
            outline_array[i, j] += 1

# Plot the binary array
plt.imshow(outline_array, cmap='binary', origin='lower', vmin=0, vmax=1, extent=[0, outline_array.shape[1], outline_array.shape[0], 0])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Binary Array - 1 (Black), 0 (White)')
plt.grid(True)
# plt.show()

# Define the base filename
base_filename = "./cavity/geometry_generated.h5"

# Check if the file already exists
suffix = 1
filename = base_filename
while os.path.exists(filename):
    suffix += 1
    filename = f"{base_filename.split('.h5')[0]}{suffix}.h5"
    
cavity_array = np.where(outline_array == 1, 0, -1) #Replace with correct material index
# Create a new HDF5 file
arr_3d = np.expand_dims(cavity_array, axis=2)
# Create a dataset within the 'data' group and store the array
with h5py.File(filename, 'w') as file:
    dset = file.create_dataset("data", data=arr_3d)

    # Add a root attribute with the name 'dx_dy_dz'
    file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)

with h5py.File(filename, 'r') as f:
    dt = f['data'][()]

# Generate the image
plt.savefig('./{}.png'.format(filename.replace('.h5','')),dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
plt.imshow(dt, cmap='viridis')

# Add a colorbar legend
cbar = plt.colorbar()
cbar.set_label('Value')



# Display the image
plt.show()

f.close()
