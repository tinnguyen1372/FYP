import numpy as np
import matplotlib.pyplot as plt
import argparse

def EnclosedCurve(r_random, N_angle, H, Center_X, Center_Y):
    ang = np.linspace(0, 2*np.pi/N_angle*(N_angle-1), N_angle)

    rho = np.random.rand(H) * 0.15 * np.logspace(-1.5, -2.5, H)
    phi = np.random.rand(H) * 2 * np.pi

    r = np.zeros(N_angle) + r_random

    for h in range(H):
        r = r + rho[h] * np.sin(h * ang + phi[h])

    x = r * np.cos(ang) + Center_X
    y = r * np.sin(ang) + Center_Y

    return x, y, r

def generate_random_array(N, mean, stddev, resolution, min_value, max_value):
    random_array = []
    while len(random_array) < N:
        random_value = np.random.normal(mean, stddev)
        rounded_value = round(random_value / resolution) * resolution
        if min_value <= rounded_value <= max_value:
            random_array.append(rounded_value)
    return random_array

def save_image(filename, x_trunk_L1, y_trunk_L1, x_trunk_L2, y_trunk_L2, x_trunk_L3, y_trunk_L3, x_Bcavity, y_Bcavity, layer1, layer2, layer3, cavity_colour):
    plt.figure(figsize=(10, 10))
    plt.fill(x_trunk_L1, y_trunk_L1, color=layer1, linestyle='none')
    plt.fill(x_trunk_L2, y_trunk_L2, color=layer2, linestyle='none')
    plt.fill(x_trunk_L3, y_trunk_L3, color=layer3, linestyle='none')
    plt.fill(x_Bcavity, y_Bcavity, color=cavity_colour, linestyle='none')
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis([-0.35, 0.35, -0.35, 0.35])
    plt.axis('off')
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_and_save_data(trunk_count, filename):
    # np.random.seed(42)  # Setting a seed for reproducibility
    # Set parameters
    Rmin_trunk = 0.15
    Rmax_trunk = 0.30
    CenTrunk_X = 0
    CenTrunk_Y = 0
    N_angle = 360
    safety_d = 0.01  # safe distance (Rmin_trunk_radius - max_cavity_radius - safety_d)
    N_trunk = trunk_count

    # Small cavity, if no need can remove
    Rmin_Scavity = 0.03
    Rmax_Scavity = 0.09

    layer1 = [1, 1, 0]
    layer2 = [1, 0.8, 0]
    layer3 = [1, 0.6, 0]
    cavity_colour = [1, 0.2, 0]

    rgbTgray = np.array([0.2989, 0.5870, 0.1140])
    layer1_G = layer1 * rgbTgray
    layer2_G = layer2 * rgbTgray
    layer3_G = layer3 * rgbTgray

    # layer4 = [1, 0.4, 0]
    # layer4_G = layer4 * rgbTgray
    layer_hole = cavity_colour * rgbTgray

    # Array to store data
    x_trunk_L1 = np.zeros((N_trunk, N_angle))
    y_trunk_L1 = np.zeros((N_trunk, N_angle))
    x_trunk_L2 = np.zeros((N_trunk, N_angle))
    y_trunk_L2 = np.zeros((N_trunk, N_angle))
    x_trunk_L3 = np.zeros((N_trunk, N_angle))
    y_trunk_L3 = np.zeros((N_trunk, N_angle))
    x_Bcavity = np.zeros((N_trunk, N_angle))
    y_Bcavity = np.zeros((N_trunk, N_angle))
    r_trunk_L3 = np.zeros((N_trunk, N_angle))
    r_trunk_L2 = np.zeros((N_trunk, N_angle))
    r_trunk_L1 = np.zeros((N_trunk, N_angle))
    r_Bcavity = np.zeros((N_trunk, N_angle))
    hole_factor = np.zeros((N_trunk,))
    R_center_Bcavity = np.zeros((N_trunk,))
    theta_Bcavity = np.zeros((N_trunk,))
    CenBcavity_X = np.zeros((N_trunk,))
    CenBcavity_Y = np.zeros((N_trunk,))

    # up: upperbound, IS: conductive layer, HW: heartwood
    up_IS = 0.9
    lo_IS = 0.75
    up_bark = 1.1
    lo_bark = 1.05

    # up_HW =0.4;
    # lo_HW=0.2;

    for i in range(N_trunk):
        # Randomize amplitude and phase.
        H = 8
        R_rand_trunk = np.random.rand(1, 1) * (Rmax_trunk - Rmin_trunk) + Rmin_trunk
        x_trunk_L2[i], y_trunk_L2[i], r_trunk_L2[i] = EnclosedCurve(
            R_rand_trunk, N_angle, 13, CenTrunk_X, CenTrunk_Y)

        ratio_L3 = np.random.rand() * (up_IS - lo_IS) + lo_IS
        x_trunk_L3[i, :] = ratio_L3 * x_trunk_L2[i, :]
        y_trunk_L3[i, :] = ratio_L3 * y_trunk_L2[i, :]
        r_trunk_L3[i, :] = ratio_L3 * r_trunk_L2[i, :]

        ratio_L1 = np.random.rand() * (up_bark - lo_bark) + lo_bark
        x_trunk_L1[i, :] = ratio_L1 * x_trunk_L2[i, :]
        y_trunk_L1[i, :] = ratio_L1 * y_trunk_L2[i, :]
        r_trunk_L1[i, :] = ratio_L1 * r_trunk_L2[i, :]

        Rmin_trunk_L3 = np.min(r_trunk_L3[i, :])
        Rmax_Bcavity = 0.8 * Rmin_trunk_L3
        Rmin_Bcavity = 0.05
        hole_factor[i] = np.random.rand(1, 1)
        R_rand_Bcavity = hole_factor[i] * (Rmax_Bcavity - Rmin_Bcavity) + Rmin_Bcavity

        min_d_Bcavity = 0.02
        max_d_Bcavity = Rmin_trunk_L3 - R_rand_Bcavity - safety_d
        R_center_Bcavity[i] = min_d_Bcavity + (max_d_Bcavity - min_d_Bcavity) * np.random.rand()
        theta_Bcavity[i] = 2 * np.pi * np.random.rand()
        CenBcavity_X[i] = R_center_Bcavity[i] * np.cos(theta_Bcavity[i])
        CenBcavity_Y[i] = R_center_Bcavity[i] * np.sin(theta_Bcavity[i])

        shape_factor = np.random.rand(1, 1)
        max_shape = 20
        min_shape = 15

        shape_cavity = np.round(shape_factor * (max_shape - min_shape) + min_shape)
        theta = np.linspace(0, 2 * np.pi, N_angle)

        x_Bcavity[i], y_Bcavity[i], r_Bcavity[i] = EnclosedCurve(
            R_rand_Bcavity, N_angle, shape_cavity, CenBcavity_X[i], CenBcavity_Y[i])

        save_image('./image/defect/defect{}.png'.format(i), x_trunk_L1[i], y_trunk_L1[i], x_trunk_L2[i], y_trunk_L2[i],
                x_trunk_L3[i], y_trunk_L3[i], x_Bcavity[i], y_Bcavity[i], layer1, layer2, layer3, cavity_colour)
        save_image('./image/healthy/healthy{}.png'.format(i), x_trunk_L1[i], y_trunk_L1[i], x_trunk_L2[i], y_trunk_L2[i],
                x_trunk_L3[i], y_trunk_L3[i], np.zeros((1, N_angle)), np.zeros((1, N_angle)), layer1, layer2, layer3, cavity_colour)

        # round the parameters to format with only 3 unit of decimal.
        range_trunk = np.zeros((4, 1))
        range_trunk[0, 0] = np.min(x_trunk_L1[i, :])
        range_trunk[1, 0] = np.max(x_trunk_L1[i, :])
        range_trunk[2, 0] = np.min(y_trunk_L1[i, :])
        range_trunk[3, 0] = np.max(y_trunk_L1[i, :])

    # hole_flag = hole_factor.copy()
    # hole_flag[hole_flag <= 0.33] = -1
    # hole_flag[(hole_flag > 0.33) & (hole_flag < 0.66)] = 0
    # hole_flag[hole_flag >= 0.66] = 1

    # height_factor = np.random.rand(1, 1000)

    np.savez('SL_trunk.npz', 
            x_trunk_L1=x_trunk_L1, 
            y_trunk_L1=y_trunk_L1, 
            x_trunk_L2=x_trunk_L2,
            y_trunk_L2=y_trunk_L2, 
            x_trunk_L3=x_trunk_L3, 
            y_trunk_L3=y_trunk_L3, 
            x_Bcavity=x_Bcavity,
            y_Bcavity=y_Bcavity, 
            range_trunk=range_trunk, 
            hole_factor=hole_factor, 
            # hole_flag=hole_flag,
            r_trunk_L1=r_trunk_L1, r_trunk_L2=r_trunk_L2, r_trunk_L3=r_trunk_L3, 
            # height_factor=height_factor,
            R_center_Bcavity=R_center_Bcavity, 
            theta_Bcavity=theta_Bcavity, 
            r_Bcavity=r_Bcavity,
            CenBcavity_X=CenBcavity_X, CenBcavity_Y=CenBcavity_Y, 
            CenTrunk_X=CenTrunk_X, CenTrunk_Y=CenTrunk_Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--trunk', type=int, help='Number of trunk')
    args = parser.parse_args()

    trunk_count = args.trunk if args.trunk is not None else 100
    filename = 'SL_trunk.npz'

    generate_and_save_data(trunk_count, filename)