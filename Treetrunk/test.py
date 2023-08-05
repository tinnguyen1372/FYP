import numpy as np

# Load the NPZ file
data = np.load('SL_trunk.npz')

# Print the keys of the data in the NPZ file
print("Keys in the NPZ file:", data.keys())

# Access and print each array in the NPZ file
for key in data.keys():
    print(f"\nData for key '{key}':")
    # print(len(data[key]))
