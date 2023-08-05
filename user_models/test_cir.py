import math

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

x1 = 0.03
y1 = 0.03
z1 = 0

x2 = 0.13
y2 = 0.03
z2 = 0

cx = 0.93
cy = 0.48
cz = 0

radius1 = math.sqrt((x1 - cx)**2 + (y1 - cy)**2 + (z1 - cz)**2)
radius2 = math.sqrt((x2 - cx)**2 + (y2 - cy)**2 + (z2 - cz)**2)

num_steps = 90

src_coord = []
rx_coord = []
src_coord.append((x1,y1,z1))
rx_coord.append((x2,y2,z2))

for i in range(0,num_steps-1):
    x1_n, y1_n, z1_n = generate_next_step_in_circular_path(x1,y1,z1,cx,cy,cz,radius1,num_steps)
    x2_n, y2_n, z2_n = generate_next_step_in_circular_path(x2,y2,z2,cx,cy,cz,radius2,num_steps)
    src_coord.append((x1_n,y1_n,z1_n))
    rx_coord.append((x2_n,y2_n,z2_n))
    x1 = x1_n
    y1 = y1_n
    z1 = z1_n

    x2 = x2_n
    y2 = y2_n
    z2 = z2_n

# with open('src_coord.txt', 'w') as f:
#    for k,v,i in src_coord:
#        f.write("{} {} {}\n".format(k,v,i))

# with open('rx_coord.txt', 'w') as f:
#    for k,v,i in rx_coord:
#        f.write("{} {} {}\n".format(k,v,i))

n = 90
for i in range(0,n):
    f = open("test_cir.in","w")
    
    f.writelines("")