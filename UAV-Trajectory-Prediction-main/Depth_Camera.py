import numpy as np
import matplotlib.pyplot as plt

# Load depth data
depth = np.load("C:\\Users\\DIAZM35\\Desktop\\UAV-Trajectory-Prediction-main\\data\\depth\\20250625-102633\\depth_step_1.npy")

# Print basic info
print("Shape:", depth.shape)
print("Min depth:", np.min(depth), "Max depth:", np.max(depth))
# Print a small 5x5 section
print(depth[60:65, 80:85])

# Visualize as image
plt.imshow(np.clip(depth, 0, 30), cmap='plasma')
plt.colorbar(label='Distance (m)')
plt.title("Depth Image - Step 1")
plt.show()