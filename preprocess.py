from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import torch


def load_center_scale_mesh(file_path: Path|str) -> o3d.cuda.pybind.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)

    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    mesh.translate(-center)

    aabb_centered = mesh.get_axis_aligned_bounding_box()
    extent = aabb_centered.get_extent()

    max_dimension = np.max(extent)
    scale_factor = 1.0 / max_dimension
    mesh.scale(scale_factor, center=np.array([0.0, 0.0, 0.0]))

    return mesh


def points_from_ground_truth(
    ground_truth_path: Path|str,
    device: str = "cuda",
    on_manifold_points_num: int = 10000,
    off_manifold_points_num: int = 20000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rescaled_ground_truth = load_center_scale_mesh(ground_truth_path)
    pc = rescaled_ground_truth.sample_points_poisson_disk(number_of_points=on_manifold_points_num)

    on_manifold_points_np = np.asarray(pc.points)
    off_manifold_points_np = np.random.uniform(-0.50, 0.50, size=(off_manifold_points_num, 3))

    on_manifold_points = torch.tensor(on_manifold_points_np, dtype=torch.float32, device=device)
    off_manifold_points = torch.tensor(off_manifold_points_np, dtype=torch.float32, device=device)

    return on_manifold_points, off_manifold_points
