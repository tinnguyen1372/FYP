import numpy as np

# Load the NPZ file
data = np.load('SL_trunk.npz')

# Print the keys of the data in the NPZ file
print("Keys in the NPZ file:", data.keys())

# Access and print each array in the NPZ file
for key in data.keys():
    print(f"\nData for key '{key}':")
    try:
        print(data[key])
    except:
        pass

import cirultils as cu
# for i in range(36):
i=0
cu.preprocess_multilayers("./Data/Defect/defect1"+ ".png", 200,"test_", i, angle = i*int(360/36))