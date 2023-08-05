import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
def generate_point_on_circle_2d(radius, x_center, y_center, z, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = x_center + radius * math.cos(angle_radians)
    y = y_center + radius * math.sin(angle_radians)
    return x, y, z

def generate_point_on_circle_3d(radius, x_center, y_center, z_center, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = x_center + radius * math.cos(angle_radians)
    y = y_center
    z = z_center + radius * math.sin(angle_radians)
    return x, y, z

def generate_next_step_in_circular_path_3d(x, y, z, cx, cy, cz, radius, num_steps):
    dx = x - cx
    dz = z - cz
    current_angle = math.atan2(dz, dx)
    angle_increment = 2 * math.pi / num_steps
    new_angle = current_angle + angle_increment
    x_new = cx + radius * math.cos(new_angle)
    y_new = y
    z_new = cz + radius * math.sin(new_angle)
    return x_new, y_new, z_new

def generate_next_step_in_circular_path(x, y, z, cx, cy, cz, radius, num_steps):
    dx = x - cx
    dy = y - cy
    current_angle = math.atan2(dy, dx)
    angle_increment = 2 * math.pi / num_steps
    new_angle = current_angle + angle_increment
    x_new = cx + radius * math.cos(new_angle)
    y_new = cy + radius * math.sin(new_angle)
    z_new = z + cz
    return x_new, y_new, z_new

def generate_points_in_circular_view(x1, y1, z1, cx, cy, cz, radius, b_scan_cnt, filename = 'cavity_coord.txt'):
    cavity_coord = []
    cavity_coord.append((x1, y1, z1))

    for i in range(b_scan_cnt - 1):
        x1_n, y1_n, z1_n = generate_next_step_in_circular_path(x1, y1, z1, cx, cy, cz, radius, b_scan_cnt)
        cavity_coord.append((x1_n, y1_n, z1_n))
        x1 = x1_n
        y1 = y1_n
        z1 = z1_n

    with open(filename, 'w') as f:
        for k, v, i in cavity_coord:
            f.write("{} {} {}\n".format(k, v, i))
    f.close()
    return cavity_coord

def generate_points_in_circular_view_3d(x1, y1, z1, cx, cy, cz, radius, b_scan_cnt, filename = 'cavity_coord.txt'):
    cavity_coord = []
    cavity_coord.append((x1, y1, z1))

    for i in range(b_scan_cnt - 1):
        x1_n, y1_n, z1_n = generate_next_step_in_circular_path_3d(x1, y1, z1, cx, cy, cz, radius, b_scan_cnt)
        cavity_coord.append((x1_n, y1_n, z1_n))
        x1 = x1_n
        y1 = y1_n
        z1 = z1_n

    with open(filename, 'w') as f:
        for k, v, i in cavity_coord:
            f.write("{} {} {}\n".format(k, v, i))
    f.close()
    return cavity_coord

def remove_coupling(filenames, src_filename = 'cir_src_only.out',bscan_num = 36):
    Ez_list = []
    # Load the data from each file
    for file_name in filenames:
        with h5py.File(file_name, 'r') as f:
            Ez = f['rxs']['rx1']['Ez'][()]
            Ez_list.append(Ez)

    with h5py.File(src_filename, 'r') as f0:
        Ez0 = f0['rxs']['rx1']['Ez'][()]

    src = Ez0[:,np.newaxis]
    Ez0 = np.repeat(src, bscan_num, axis=1)

    # Compute the differences relative to the first file
    Ez_diff_list = [np.subtract(Ez, Ez0) for Ez in Ez_list]

    Ez = np.concatenate(Ez_diff_list, axis=1)

    return Ez

def remove_coupling_3d(filenames, src_filename = 'cir_src_only_3d.out',bscan_num = 36):
    Ey_list = []
    # Load the data from each file
    for file_name in filenames:
        with h5py.File(file_name, 'r') as f:
            Ey = f['rxs']['rx1']['Ey'][()]
            if file_name == 'healthy_3d.out':
                Ey = np.repeat(Ey, bscan_num, axis=1)
            Ey_list.append(Ey)


    with h5py.File(src_filename, 'r') as f0:
        Ey0 = f0['rxs']['rx1']['Ey'][()]

    src = Ey0[:,np.newaxis]
    print(np.shape(src))
    # Ey0 = np.repeat(src, bscan_num, axis=1)
    print(np.shape(Ey0))
    # Compute the differences relative to the first file
    Ey_diff_list = [np.subtract(Ey, Ey0) for Ey in Ey_list]

    Ey = np.concatenate(Ey_diff_list, axis=1)

    return Ey


def EnclosedCurve(r_random, N_angle, H, Center_X, Center_Y):
    ang = np.linspace(0, 2*np.pi/N_angle*(N_angle-1), N_angle)
    
    rho = np.round(np.random.rand(int(H)) * 0.15 * np.logspace(-1.5, -2.5, int(H)))
    phi = np.random.rand(int(H)) * 2*np.pi
    
    r = np.zeros(N_angle) + r_random
    
    for h in range(int(H)):
        r = r + rho[h]*np.sin(h*ang + phi[h])
    
    x = r * np.cos(ang) + Center_X
    y = r * np.sin(ang) + Center_Y
    
    return x, y, r

def preprocess(image_path, res,prefix):
    # image_path = "TreeGen/image/defect/defect0.png"
    res = int(res)
    img = Image.open(image_path).convert('RGB')

    # Resize the image to 200x200
    img_resized = img.resize((res, res))

    # Convert the resized image to a 2D array of integers
    color_map = {
        (255, 255, 255): -1,  # White
        (255, 255, 0): 0,  # Yellow
        (255, 51, 0): 1  # Red
    }

    arr_2d = np.zeros((res, res), dtype=int)
    for y in range(res):
        for x in range(res):
            pixel_color = img_resized.getpixel((x, y))
            arr_2d[y, x] = color_map.get(pixel_color, 0)

    # Expand dimensions
    arr_3d = np.expand_dims(arr_2d, axis=2)

    base_filename = ""
    if "healthy" in image_path:
        base_filename = os.getcwd() + '/' + prefix + 'healthy.h5'
    else:
        base_filename = os.getcwd() + '/' + prefix + 'cavity.h5'

    filename = base_filename
    with h5py.File(filename, 'w') as file:
        dset = file.create_dataset("data", data=arr_3d)
        file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)

def increment_file_index(string):
        result = ""
        index_start = -1
        index_end = -1

        for i, char in enumerate(string):
            if char.isdigit():
                if index_start == -1:
                    index_start = i
                index_end = i
            elif index_start != -1:
                break

        if index_start == -1:
            return string

        index = int(string[index_start:index_end + 1])
        next_index = index + 1
        replaced_index = str(next_index)
        result = string[:index_start] + replaced_index + string[index_end + 1:]
        return result
