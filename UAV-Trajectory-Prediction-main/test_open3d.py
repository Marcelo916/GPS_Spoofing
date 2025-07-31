import open3d as o3d
import numpy as np
import time

# Generate dummy point cloud
points = np.random.rand(1000, 3) * 10
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualizer setup
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Lidar Live Test")
vis.add_geometry(pcd)

for i in range(100):
    points = np.random.rand(1000, 3) * 10
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)

vis.destroy_window()
