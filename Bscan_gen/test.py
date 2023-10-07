from cirultils import preprocess_multilayers_3d

# image_path = 'test.png'
# preprocess_multilayers_3d(image_path, 200, 'test_', 1, angle = 0)

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the HDF5 file containing 3D data
file_path = 'FYP_healthy.h5'
f = h5py.File(file_path, 'r')

# Access the 3D data from the HDF5 file
dset = f['data']

# Generate the image
fig, ax = plt.subplots(1, 3,figsize=(10, 5))
ax[0].imshow(np.transpose(dset[:,125,:], axes=(1, 0)), cmap='viridis')
ax[0].invert_yaxis()
ax[0].set_title('Domain View')

# # Plot dset2 on the right subplot
ax[1].imshow(np.transpose(dset[:,:,36], axes=(1, 0)), cmap='viridis')
ax[1].invert_yaxis()
ax[1].set_title('Trunk View')

ax[2].imshow(np.transpose(dset[:,:,154], axes=(1, 0)), cmap='viridis')
ax[2].invert_yaxis()
ax[2].set_title('Antenna View')


plt.show()
# Close the HDF5 file
f.close()

