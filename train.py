import numpy as np
import open3d as o3d
import torch

from model import Siren, train_siren_on_points, reconstruct_mesh
from preprocess import load_center_scale_mesh
from dataclasses import dataclass
from enum import Enum

class Model(Enum):
    SIMPLE = Siren

@dataclass
class Config:
    model: Model = Model.SIMPLE

    gt_path: str = "/home/mikolaj/Documents/github/inr_voronoi/data/00000003/00000003_ground_truth.obj"
    on_manifold_points_num: int = 10000
    off_manifold_points_num: int = 20000
    reconstruction_resolution: int = 256

    in_features: int = 3
    hidden_features: int = 256
    hidden_layers: int = 5
    out_features: int = 1

    epochs: int = 500

    device = "cuda"

def train_single(config: Config) -> o3d.geometry.TriangleMesh:
    rescaled_ground_truth = load_center_scale_mesh(config.gt_path)
    pc = rescaled_ground_truth.sample_points_poisson_disk(number_of_points=config.on_manifold_points_num)

    on_manifold_points_np = np.asarray(pc.points)
    off_manifold_points_np = np.random.uniform(-0.50, 0.50, size=(config.off_manifold_points_num, 3))

    on_manifold_points = torch.tensor(on_manifold_points_np, dtype=torch.float32, device=config.device)
    off_manifold_points = torch.tensor(off_manifold_points_np, dtype=torch.float32, device=config.device)

    model = config.model.value(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        out_features=config.out_features,
    )
    model.to(config.device)

    train_siren_on_points(model, on_manifold_points, off_manifold_points, epochs=config.epochs, verbose=True)

    reconstructed_mesh = reconstruct_mesh(model, resolution=config.reconstruction_resolution, device=config.device)
    return reconstructed_mesh

if __name__ == "__main__":
    config = Config()

    reconstruction = train_single(config)

    o3d.visualization.draw_geometries([reconstruction])