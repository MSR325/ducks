import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .npz file
data = np.load('video_droid.npz')
cam_c2w = data['cam_c2w']  # shape (N, 4, 4)

# Print shape
print("cam_c2w shape:", cam_c2w.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scale = 0.05  # Length of each axis (tweak as needed)

for pose in cam_c2w:
    # Extract translation
    t = pose[:3, 3]

    # Extract rotation (columns are the world-frame axes of the camera)
    x_axis = pose[:3, 0] * scale
    y_axis = pose[:3, 1] * scale
    z_axis = pose[:3, 2] * scale

    # Plot camera origin
    ax.scatter(t[0], t[1], t[2], c='k', s=10)

    # Plot camera axes
    ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color='r')
    ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color='g')
    ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera-to-World Transformations')
plt.show()
