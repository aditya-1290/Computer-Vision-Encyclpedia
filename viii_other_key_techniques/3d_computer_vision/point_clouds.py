"""
Point Clouds: 3D representation using point sets.

Implementation uses Open3D for basic point cloud operations.

Theory:
- Point cloud: Set of 3D points (x,y,z) with optional attributes (color, normals).
- Applications: 3D reconstruction, SLAM, autonomous driving.

Math: Point cloud P = {(x_i, y_i, z_i) | i=1 to N}

Reference:
- Open3D library documentation
"""

import numpy as np
import open3d as o3d

def create_point_cloud(points, colors=None):
    """
    Create a point cloud from points and optional colors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_point_cloud(pcd):
    """
    Visualize the point cloud.
    """
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Create a simple point cloud
    points = np.random.rand(1000, 3)
    colors = np.random.rand(1000, 3)
    pcd = create_point_cloud(points, colors)
    print(f"Point cloud has {len(pcd.points)} points")
    # visualize_point_cloud(pcd)  # Uncomment to visualize
