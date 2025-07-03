import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int,
                 outermost_linear: bool = True, first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


def train_siren_on_points(model: Siren, on_manifold_points: torch.Tensor, off_manifold_points: torch.Tensor,
                          epochs: int = 1000, lr: float = 1e-4) -> None:
    model.train()
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    all_points = torch.cat([on_manifold_points, off_manifold_points], dim=0)
    num_on_mani_points = on_manifold_points.shape[0]

    for _ in tqdm(range(epochs), desc=f"Training part model", leave=False):
        optim.zero_grad()
        model_out, model_in = model(all_points)
        total_loss, _, _, _ = sdf_siren_loss(model_out, model_in, num_on_mani_points)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()


def reconstruct_mesh(model: Siren, resolution: int = 128, device: str = 'cpu') -> o3d.geometry.TriangleMesh:
    model.to(device)
    model.eval()

    grid_vals = torch.linspace(-0.5, 0.5, resolution)
    x, y, z = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij')
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(device)

    sdf_values = []
    with torch.no_grad():
        for p in tqdm(torch.split(points, 10000, dim=0), desc="Reconstructing", leave=False):
            sdf_values.append(model(p)[0].cpu())
    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape(resolution, resolution, resolution)

    from skimage import measure

    spacing_val = 1.0 / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(sdf_values, level=0.0, spacing=(spacing_val, spacing_val, spacing_val))

    verts -= 0.5

    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(verts)
    reconstructed_mesh.triangles = o3d.utility.Vector3iVector(faces)
    reconstructed_mesh.compute_vertex_normals()

    return reconstructed_mesh


def sdf_siren_loss(model_output: torch.Tensor, coords: torch.Tensor, num_on_mani: int,
                   alpha: float = 100.0, lambda_on: float = 1.0, lambda_dnm: float = 0.1, lambda_eik: float = 0.1) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    on_manifold_pred = model_output[:num_on_mani]
    off_manifold_pred = model_output[num_on_mani:]

    on_manifold_loss = torch.abs(on_manifold_pred).mean() if num_on_mani > 0 else torch.tensor(0.0)

    off_manifold_loss = torch.exp(-alpha * torch.abs(off_manifold_pred)).mean() if off_manifold_pred.shape[
                                                                                       0] > 0 else torch.tensor(0.0)

    d_output = torch.ones_like(model_output, requires_grad=False, device=model_output.device)
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=coords,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    eikonal_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    total_loss = lambda_on * on_manifold_loss + lambda_dnm * off_manifold_loss + lambda_eik * eikonal_loss

    return total_loss, on_manifold_loss, off_manifold_loss, eikonal_loss


if __name__ == '__main__':
    part_name = "00000003"
    root_path = Path(f"/home/mikolaj/Documents/github/inr-voronoi/data/{part_name}")

    mesh = o3d.io.read_triangle_mesh(str(root_path / "gt_rescaled.ply"))
    mesh_name = part_name

    parts_dir = root_path / "parts"

    part_mesh_paths = sorted(list(parts_dir.glob("*.ply")))

    print(f"Found {len(part_mesh_paths)} mesh parts.")
    part_meshes = [o3d.io.read_triangle_mesh(str(p)) for p in part_mesh_paths]

    print("Building KDTrees for each mesh part...")
    part_kdtrees = [o3d.geometry.KDTreeFlann(p) for p in part_meshes]

    print("Sampling on- and off-manifold points...")
    num_on_mani_points = 10000
    num_off_mani_points = 20000

    pcd = mesh.sample_points_poisson_disk(number_of_points=num_on_mani_points, init_factor=5)
    on_manifold_points_np = np.asarray(pcd.points)
    off_manifold_points_np = np.random.uniform(-0.5, 0.5, size=(num_off_mani_points, 3))

    global_off_manifold_tensor = torch.tensor(off_manifold_points_np, dtype=torch.float32)

    print("Assigning on-manifold points to the closest mesh part...")
    on_manifold_part_batches: Dict[int, List] = defaultdict(list)

    for point in tqdm(on_manifold_points_np, desc="Assigning on-manifold points"):
        distances = []
        for kdtree in part_kdtrees:
            _, _, dist2 = kdtree.search_knn_vector_3d(point, 1)
            distances.append(dist2[0])
        closest_part_idx = np.argmin(distances)
        on_manifold_part_batches[closest_part_idx].append(point)

    output_dir = Path("outputs")

    for part_idx in tqdm(range(len(part_meshes)), desc="Processing Parts"):
        part_on_points = torch.tensor(np.array(on_manifold_part_batches.get(part_idx, [])), dtype=torch.float32)

        if part_on_points.shape[0] == 0:
            tqdm.write(f"Skipping part {part_idx} (no on-manifold points assigned).")
            continue

        tqdm.write(f"\n--- Processing Part {part_idx} ---")
        tqdm.write(
            f"On-manifold points: {part_on_points.shape[0]}, Off-manifold points: {global_off_manifold_tensor.shape[0]}")

        part_model = Siren(in_features=3, out_features=1, hidden_features=256,
                           hidden_layers=5, outermost_linear=True)

        train_siren_on_points(model=part_model, on_manifold_points=part_on_points,
                              off_manifold_points=global_off_manifold_tensor, epochs=500)

        tqdm.write(f"Reconstructing mesh for part {part_idx}...")
        reconstructed_mesh = reconstruct_mesh(model=part_model, resolution=128)

        if len(reconstructed_mesh.triangles) > 0:
            reconstructed_mesh_path = output_dir / f"{mesh_name}_part_{part_idx}.ply"
            output_dir.mkdir(parents=True, exist_ok=True)
            tqdm.write(f"Saving reconstructed mesh for part {part_idx} to {reconstructed_mesh_path}")
            o3d.io.write_triangle_mesh(str(reconstructed_mesh_path), reconstructed_mesh)
        else:
            tqdm.write(f"Skipping save for part {part_idx} as reconstruction was empty.")

    print("\nAll parts processed.")
