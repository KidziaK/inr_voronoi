from typing import Tuple
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from skimage import measure
from tqdm import tqdm


class SineLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            is_first: bool = False,
            omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            hidden_layers: int,
            out_features: int,
            outermost_linear: bool = True,
            first_omega_0: float = 30.0,
            hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


def train_siren_on_points(
        model: Siren,
        on_manifold_points: torch.Tensor,
        off_manifold_points: torch.Tensor,
        epochs: int = 1000,
        lr: float = 1e-4,
        verbose: bool = False,
) -> None:
    model.train()
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    all_points = torch.cat([on_manifold_points, off_manifold_points], dim=0)
    num_on_mani_points = on_manifold_points.shape[0]

    with tqdm(range(epochs), desc=f"Training SIREN", leave=False) as pbar:
        for epoch in pbar:
            optim.zero_grad()
            model_out, model_in = model(all_points)
            total_loss, _, _, _ = sdf_siren_loss(model_out, model_in, num_on_mani_points)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            pbar.set_description(f"Epoch = {epoch} | Total loss = {total_loss:.04f}")


def reconstruct_mesh(
        model: Siren,
        aabb: o3d.cuda.pybind.geometry.AxisAlignedBoundingBox,
        resolution: int = 128,
        device: str = "cpu"
) -> o3d.geometry.TriangleMesh:
    model.to(device)
    model.eval()

    x_vals = torch.linspace(aabb.min_bound[0], aabb.max_bound[0], resolution)
    y_vals = torch.linspace(aabb.min_bound[1], aabb.max_bound[1], resolution)
    z_vals = torch.linspace(aabb.min_bound[2], aabb.max_bound[2], resolution)

    x, y, z = torch.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(device)

    sdf_values = []
    with torch.no_grad():
        for p in tqdm(
                torch.split(points, 10000, dim=0), desc="Reconstructing", leave=False
        ):
            sdf_values.append(model(p)[0].cpu())
    sdf_values = (
        torch.cat(sdf_values, dim=0).numpy().reshape(resolution, resolution, resolution)
    )

    verts, faces, _, _ = measure.marching_cubes(
        sdf_values, level=0.0
    )

    verts_normalized = verts / (resolution - 1)

    aabb_min = aabb.min_bound
    aabb_max = aabb.max_bound
    side_lengths = aabb_max - aabb_min
    verts_world = verts_normalized * side_lengths + aabb_min

    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    reconstructed_mesh.triangles = o3d.utility.Vector3iVector(faces)
    reconstructed_mesh.compute_vertex_normals()

    return reconstructed_mesh


def sdf_siren_loss(
        model_output: torch.Tensor,
        coords: torch.Tensor,
        num_on_mani: int,
        alpha: float = 100.0,
        lambda_on: float = 1.0,
        lambda_dnm: float = 0.1,
        lambda_eik: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    on_manifold_pred = model_output[:num_on_mani]
    off_manifold_pred = model_output[num_on_mani:]

    on_manifold_loss = (
        torch.abs(on_manifold_pred).mean() if num_on_mani > 0 else torch.tensor(0.0)
    )

    off_manifold_loss = (
        torch.exp(-alpha * torch.abs(off_manifold_pred)).mean()
        if off_manifold_pred.shape[0] > 0
        else torch.tensor(0.0)
    )

    d_output = torch.ones_like(
        model_output, requires_grad=False, device=model_output.device
    )
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=coords,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    eikonal_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    total_loss = (
            lambda_on * on_manifold_loss
            + lambda_dnm * off_manifold_loss
            + lambda_eik * eikonal_loss
    )

    return total_loss, on_manifold_loss, off_manifold_loss, eikonal_loss
