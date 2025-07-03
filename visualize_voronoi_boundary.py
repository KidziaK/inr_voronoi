import numpy as np
import open3d as o3d

if __name__ == "__main__":
    boundaries_path = "/home/mikolaj/Documents/github/inr-voronoi/data/00000003/voronoi_boundaries.npy"
    features_path = "/home/mikolaj/Documents/github/inr-voronoi/data/00000003/features.npy"

    boundaries = np.load(boundaries_path)

    indices = (np.argwhere(boundaries != 0).astype(np.float32) - 128) / 128
    shape = np.array(boundaries.shape)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indices)

    original_mesh = o3d.io.read_triangle_mesh(f"/home/mikolaj/Documents/github/inr-voronoi/data/00000003/gt_rescaled.ply")

    o3d.visualization.draw([original_mesh, pcd])
