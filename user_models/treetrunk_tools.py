from cmath import sqrt
from gprMax.gprMax import api


def domain_tool(radius, distance, b_scan, highest_permitivity,z = 0, src_receive_dist = 0.15,pml_redundancy = 0.002*10, light_speed = 299792458, buffer = 0.02):
    '''function calculate the params base on radius, distance to trunks and number of bscan, return in order of domain, trunk_center, src, rx, steps, and min window'''
    diameter = radius*2
    sharp_domain = [(radius*2)*3, radius*2+distance]
    total_redundancy = buffer/2 + pml_redundancy # 0.03
    redundant_domain = (sharp_domain[0] + total_redundancy*2, sharp_domain[1] +  total_redundancy*2) #0.96,0.56
    trunk_center = (diameter+radius+total_redundancy, distance+radius+total_redundancy) #0.48,0.38
    src_position = (total_redundancy, total_redundancy , z) #0.03, 0.03, 0
    receive_position = (total_redundancy+0.1, total_redundancy, z) #0.13,0.03,0
    step_length = (sharp_domain[0] - src_receive_dist)/b_scan
    min_time_window = (sqrt(pow(sharp_domain[0], 2) + pow(sharp_domain[1], 2)))*2/((light_speed)/highest_permitivity)

    return redundant_domain, total_redundancy, trunk_center, src_position,receive_position,step_length,min_time_window

def src_only_generator(domain_x,domain_y,time_window,frequency,domain_z = None,freq_type=None,freq_placeholder=None,dx=None,dy=None,dz=None):
    '''Generate a a-scan of a scource only based on domain, time_window[e-9] and 
    frequency[e9], type[default ricker], and placeholder[default my_wave], dx_dy_dz[default 0.002]'''
    with open("src_only.in","w") as f:
        if dx == None:
            dx = 0.002
        if dy == None:
            dy = 0.002
        if dz == None:
            dz = 0.002
        if domain_z == None:
            domain_z = 0.002
        if freq_type == None:
            freq_type = 'ricker'
        if freq_placeholder == None :
            freq_placeholder = 'my_wave'
        f.write("#title: Source Only\n")
        f.write("#domain: {} {} {}\n".format(domain_x,domain_y,domain_z))
        f.write("#dx_dy_dz: {} {} {}\n".format(dx,dy,dz))
        f.write("#time_window: {}e-9\n".format(time_window))
        f.write("#waveform: {} 1 {}e9 {}\n".format(freq_type,frequency,freq_placeholder))
    api("src_only.in", n=1, geometry_only=False)

def img_deduction(src_file,merged_file,b_scan):
    '''deduct the coupling in image with src_file, merged file and b_scan number'''
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    file_name = merged_file
    Ez_list = []

    # Load the data from each file
    with h5py.File(file_name, 'r') as f:
        Ez = f['rxs']['rx1']['Ez'][()]
        Ez_list.append(Ez)

    with h5py.File(src_file, 'r') as f0:
        Ez0 = f0['rxs']['rx1']['Ez'][()]

    # src = Ez0[:, 0]
    src = Ez0[:, np.newaxis]  # Add a new axis

    Ez0 = np.repeat(src, b_scan , axis=1)

    # Compute the differences relative to the first file
    Ez_diff_list = [np.subtract(Ez, Ez0) for Ez in Ez_list]

    Ez = np.concatenate(Ez_diff_list, axis=1)

    plt.imshow(Ez, cmap='seismic', aspect='auto')

    return

import math

def generate_circular_path_coordinates(x1, y1, z1, x2, y2, z2, cx, cy, cz, num_steps):
    """
    Generates source and receiver coordinates for a circular path.
    Parameters:
        x1, y1, z1: Starting coordinates of the source.
        x2, y2, z2: Starting coordinates of the receiver.
        cx, cy, cz: Center of the circular path.
        num_steps: Total number of steps in the circular path.
    Returns:
        Two lists of tuples representing the source and receiver coordinates.
    """

    def generate_next_step_in_circular_path(x, y, z, cx, cy, cz, radius, num_steps):
        """
        Generates the next step in coordinates for a point (x, y, z) moving in a circular path.
        Parameters:
            x, y, z: Current coordinates of the point.
            cx, cy, cz: Center of the circular path.
            radius: Radius of the circular path.
            num_steps: Total number of steps in the circular path.
        Returns:
            A list of tuples representing the new coordinates [(x1, y1, z1)].
        """

        # Calculate the current angle based on the current coordinates
        dx = x - cx
        dy = y - cy
        current_angle = math.atan2(dy, dx)

        # Calculate the angle increment per step
        angle_increment = 2 * math.pi / num_steps

        # Calculate the new angle for the next step
        new_angle = current_angle + angle_increment

        # Calculate the new coordinates using the new angle and radius
        x_new = cx + radius * math.cos(new_angle)
        y_new = cy + radius * math.sin(new_angle)
        z_new = z + cz  # Assuming no change in the z-coordinate
        
        

        return x_new, y_new, z_new
    radius1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2 + (z1 - cz)**2)
    radius2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2 + (z2 - cz)**2)

    src_coord = [(x1, y1, z1)]
    rx_coord = [(x2, y2, z2)]

    for i in range(num_steps - 1):
        x1_n, y1_n, z1_n = generate_next_step_in_circular_path(x1, y1, z1, cx, cy, cz, radius1, num_steps)
        x2_n, y2_n, z2_n = generate_next_step_in_circular_path(x2, y2, z2, cx, cy, cz, radius2, num_steps)
        src_coord.append((x1_n, y1_n, z1_n))
        rx_coord.append((x2_n, y2_n, z2_n))
        x1, y1, z1 = x1_n, y1_n, z1_n
        x2, y2, z2 = x2_n, y2_n, z2_n

    return src_coord, rx_coord