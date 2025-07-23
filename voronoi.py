import numba
import numpy as np
import open3d as o3d
import torch

from typing import List
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


@numba.njit(parallel=True)
def find_sharp_edges(points: np.ndarray, normals: np.ndarray, neighbors: np.ndarray, angle_threshold_rad: float) -> np.ndarray:
    n_points = len(points)
    is_edge = np.zeros(n_points, dtype=numba.boolean)

    for i in numba.prange(n_points):
        ni = normals[i]

        for j_idx in range(1, neighbors.shape[1]):
            j = neighbors[i, j_idx]
            nj = normals[j]

            dot_product = np.abs(np.dot(ni, nj))

            angle = np.arccos(dot_product)

            if angle > angle_threshold_rad:
                is_edge[i] = True
                break

    return is_edge


def find_connected_components(nodes, adjacency_matrix):
    n = len(nodes)
    parent = np.arange(n)

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] != 0:
                union(i, j)

    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    result = [nodes[indices] for indices in components.values() if len(indices) > 50]
    return result


def fit_plane_to_points(points: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3)
    pca.fit(points)

    pca_normal = pca.components_[-1]
    centroid = pca.mean_

    d = -np.dot(pca_normal, centroid)
    return np.append(pca_normal, d)


def merge_similar_planes(planes: List[np.ndarray], normal_threshold: float = 0.99, d_threshold: float = 0.05) -> np.ndarray:
    merged_planes = []
    processed_indices = np.zeros(len(planes), dtype=bool)

    for i in range(len(planes)):
        if processed_indices[i]:
            continue

        current_plane = planes[i]
        similar_planes_group = [current_plane]
        processed_indices[i] = True

        for j in range(i + 1, len(planes)):
            if processed_indices[j]:
                continue

            other_plane = planes[j]

            dot_product = np.abs(np.dot(current_plane[:3], other_plane[:3]))

            dist_diff = np.abs(current_plane[3] - other_plane[3])

            if dot_product > normal_threshold and dist_diff < d_threshold:
                similar_planes_group.append(other_plane)
                processed_indices[j] = True

        merged_plane = np.mean(similar_planes_group, axis=0)
        merged_planes.append(merged_plane)

    return np.array(merged_planes)

def voronoi_from_points(
    points: torch.Tensor,
    k_for_normals: int = 5,
    k_for_edges: int = 10,
    angle_threshold_degrees: float = 35.0,
    visualize: bool = False,
    verbose: bool = False
) -> List[torch.Tensor]:
    points_np = points.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_for_normals))
    pcd.orient_normals_consistent_tangent_plane(k=k_for_normals)
    normals = np.asarray(pcd.normals)

    kdtree = cKDTree(points_np)
    _, neighbors = kdtree.query(points_np, k=k_for_edges)
    angle_threshold_rad = np.deg2rad(angle_threshold_degrees)
    is_edge_mask = find_sharp_edges(points_np, normals, neighbors, angle_threshold_rad)
    edge_points = points_np[is_edge_mask]

    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    edge_kdtree = cKDTree(edge_points)

    if visualize:
        o3d.visualization.draw_geometries([edge_pcd])

    distances, _ = edge_kdtree.query(edge_points, k=5)
    adjecency_radius = distances.max(axis=1).mean()

    condensed_distances = pdist(edge_points, 'euclidean')
    distance_matrix = squareform(condensed_distances)

    adjecency_matrix = np.where(distance_matrix < adjecency_radius, True, False).astype(np.bool)

    components = find_connected_components(edge_points, adjecency_matrix)

    fitted_planes = []
    for c in components:
        plane_eq = fit_plane_to_points(c)
        fitted_planes.append(plane_eq)

    fitted_planes = merge_similar_planes(fitted_planes)

    if len(fitted_planes) == 0:
        return [points]

    plane_normals = fitted_planes[:, :3]
    plane_offsets = fitted_planes[:, 3]

    signed_distances = np.einsum('ij,kj->ik', points_np, plane_normals) + plane_offsets
    side_of_plane_matrix = (signed_distances > 0)
    cell_labels = side_of_plane_matrix.dot(1 << np.arange(side_of_plane_matrix.shape[-1] - 1, -1, -1))

    unique_cell_labels = np.unique(cell_labels)

    voronoi_cells = []
    for i, label in enumerate(unique_cell_labels):
        cell_mask = (cell_labels == label)
        cell_points = points_np[cell_mask]

        cell_kdtree = cKDTree(cell_points)

        distances, _ = cell_kdtree.query(cell_points, k=5)
        adjecency_radius = distances.max(axis=1).mean()

        condensed_distances = pdist(cell_points, 'euclidean')
        distance_matrix = squareform(condensed_distances)

        adjecency_matrix = np.where(distance_matrix < adjecency_radius, True, False).astype(np.bool)

        components = find_connected_components(cell_points, adjecency_matrix)

        for c in components:
            voronoi_cells.append(torch.tensor(c, dtype=torch.float32, device="cuda"))

    if verbose:
        print(f"Extracted {len(voronoi_cells)} voronoi cells")

    return voronoi_cells
