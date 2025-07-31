import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Lidar data from text file (adjust the path as needed)
lidar_file = r"C:\Users\DIAZM35\Desktop\UAV-Trajectory-Prediction-main\data\lidar\20250625-102633\lidar_step_27.txt"

# Load data into numpy array
points = np.loadtxt(lidar_file, delimiter=',')

# Check shape (should be [N, 3])
print("Point cloud shape:", points.shape)

# Extract x, y, z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Plot 3D point cloud
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=3, c=z, cmap='viridis')  # color by height (z)
print("Total Lidar points:", len(x))

ax.set_title("Lidar Point Cloud - Step 1")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=1, alpha=0.5)
plt.title("Top-down Lidar View (X-Y plane)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid(True)
plt.show()
