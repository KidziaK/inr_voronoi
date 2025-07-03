from typing import Tuple

import open3d as o3d
import numpy as np

from pathlib import Path

from voronoi_extraction.extract import extract_voronoi_from_single_ply, voronoi_boundary_to_voronoi_cell

def load_center_scale_mesh(file_path: Path) -> o3d.cuda.pybind.geometry.TriangleMesh:
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


def full_preprocess(root_path: Path, num_points: int = 10000) -> None:
    gt_path = root_path / "gt.ply"

    preprocessed_abc_mesh = load_center_scale_mesh(gt_path)
    preprocessed_abc_mesh_path = root_path / "gt_rescaled.ply"
    o3d.io.write_triangle_mesh(preprocessed_abc_mesh_path, preprocessed_abc_mesh)

    input_point_cloud = preprocessed_abc_mesh.sample_points_poisson_disk(number_of_points=num_points)
    input_point_cloud_path = root_path / "input.ply"
    o3d.io.write_point_cloud(input_point_cloud_path, input_point_cloud)

    extract_voronoi_from_single_ply(input_point_cloud_path)
    vornoi_boundaries_path = root_path / "voronoi_boundaries.npy"
    features_path = root_path / "voronoi_feat.npy"

    voronoi_boundary_to_voronoi_cell(vornoi_boundaries_path, features_path)


if __name__ == "__main__":
    part_name = "00000003"
    root_path = Path(f"/home/mikolaj/Documents/github/inr-voronoi/data/{part_name}")
    full_preprocess(root_path)
