import numpy as np
import open3d as o3d
import torch

from model import Siren, train_siren_on_points, reconstruct_mesh
from preprocess import points_from_ground_truth
from voronoi import voronoi_from_points

Mesh = o3d.geometry.TriangleMesh

def train_single_and_reconstruct(on_mani_pts: torch.Tensor, epochs: int = 500) -> Mesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(on_mani_pts.cpu().numpy())
    aabb = pcd.get_axis_aligned_bounding_box()

    n = len(on_mani_pts)

    padding = (aabb.max_bound - aabb.min_bound).max()

    off_mani_points_np = np.random.uniform(aabb.min_bound - padding , aabb.max_bound + padding, size=(2 * n, 3))
    off_mani_points = torch.tensor(off_mani_points_np, dtype=torch.float32, device="cuda")

    model = Siren(
        in_features=3,
        hidden_features=256,
        hidden_layers=5,
        out_features=1,
    )
    model.to("cuda")

    train_siren_on_points(model, on_mani_pts, off_mani_points, epochs=epochs, verbose=True)



    reconstructed_mesh = reconstruct_mesh(model, aabb, resolution=256, device="cuda")
    return reconstructed_mesh


if __name__ == "__main__":
    o3d.utility.random.seed(69)

    gt_path: str = "/home/mikolaj/Documents/github/inr_voronoi/data/00000003/00000003_ground_truth.obj"

    on_manifold_points, off_manifold_points = points_from_ground_truth(gt_path)
    off_manifold_points_np = off_manifold_points.cpu().numpy()

    voronoi_cells = voronoi_from_points(on_manifold_points, visualize=False, verbose=True)

    for i, on_pts in enumerate(voronoi_cells):
        reconstruction = train_single_and_reconstruct(on_pts, epochs=10000)
        o3d.visualization.draw_geometries([reconstruction])
        o3d.io.write_triangle_mesh(f"/home/mikolaj/Documents/github/inr_voronoi/reconstructions/{i:02d}.ply", reconstruction)
