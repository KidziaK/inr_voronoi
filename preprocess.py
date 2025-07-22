from pathlib import Path

import numpy as np
import open3d as o3d


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
