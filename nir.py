import os
import sys
import math
import time
import logging
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Configure logging for R&D traceability
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')
logger = logging.getLogger("NIR_Master_Pipeline")

# ==========================================
# 1. MODEL ARCHITECTURE (SIREN)
# ==========================================
class SineActivation(nn.Module):
    """Periodic activation function to retain high-frequency spatial-spectral details."""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SIRENNIRRegressor(nn.Module):
    """Continuous Neural Implicit Representation (NIR) network mapping coordinates to intensities."""
    def __init__(self, in_features: int = 3, out_features: int = 1, 
                 hidden_features: int = 128, hidden_layers: int = 4, 
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        self.net = []
        
        # First Layer initialization
        first_layer = nn.Linear(in_features, hidden_features)
        self.init_siren_layer(first_layer, is_first=True, omega_0=first_omega_0)
        self.net.append(first_layer)
        self.net.append(SineActivation(omega_0=first_omega_0))
        
        # Hidden Layers
        for _ in range(hidden_layers):
            layer = nn.Linear(hidden_features, hidden_features)
            self.init_siren_layer(layer, is_first=False, omega_0=hidden_omega_0)
            self.net.append(layer)
            self.net.append(SineActivation(omega_0=hidden_omega_0))
            
        # Final Output Layer
        final_layer = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            num_input = final_layer.in_features
            final_layer.weight.data.uniform_(-math.sqrt(6 / num_input) / hidden_omega_0, 
                                              math.sqrt(6 / num_input) / hidden_omega_0)
            if final_layer.bias is not None:
                final_layer.bias.data.zero_()
        self.net.append(final_layer)
        
        self.net = nn.Sequential(*self.net)

    @staticmethod
    def init_siren_layer(layer: nn.Linear, is_first: bool, omega_0: float) -> None:
        """Applies exact SIREN initialization scheme to avoid vanishing gradients."""
        with torch.no_grad():
            num_input = layer.in_features
            bounds = 1 / num_input if is_first else math.sqrt(6 / num_input) / omega_0
            layer.weight.data.uniform_(-bounds, bounds)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


# ==========================================
# 2. DATA PREPROCESSING MODULE
# ==========================================
class HyperspectralPreprocessor:
    """Transforms standard discrete spatial grids into continuous coordinate spaces."""
    @staticmethod
    def generate_training_data(data_cube: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        try:
            height, width, bands = data_cube.shape
            metadata = {"height": height, "width": width, "bands": bands}
            
            # Linearly map dimensions perfectly into range [-1, 1]
            x_coords = torch.linspace(-1, 1, steps=height)
            y_coords = torch.linspace(-1, 1, steps=width)
            lambda_coords = torch.linspace(-1, 1, steps=bands)
            
            grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, lambda_coords, indexing='ij')
            
            # Flatten to continuous 3D coordinate pairs [N, 3]
            coords_flat = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)
            targets_flat = torch.tensor(data_cube, dtype=torch.float32).view(-1, 1).to(device)
            
            return coords_flat, targets_flat, metadata
        except Exception as e:
            logger.error(f"Failed to generate coordinate training bounds: {str(e)}")
            sys.exit(1)


# ==========================================
# 3. OPTIMIZATION LOOP
# ==========================================
def train_nir_pipeline(model: nn.Module, coords: torch.Tensor, targets: torch.Tensor, 
                      epochs: int = 50, lr: float = 1e-3, batch_size: int = 16384) -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset_size = coords.shape[0]
    
    logger.info(f"Initiating NIR Optimization Engine ({epochs} Epochs)...")
    model.train()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i: i + batch_size]
            batch_coords, batch_targets = coords[indices], targets[indices]
            
            predictions = model(batch_coords)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{epochs} | Step Loss (MSE): {epoch_loss / batches:.6f}")
            
    return time.time() - start_time


# ==========================================
# 4. DECOMPRESSION & AEROSPACE EVALUATION SUITE
# ==========================================
class EvaluationDecompressionSuite:
    """Handles weight serialization, footprint calculations, and precision telemetry."""
    @staticmethod
    def evaluate_and_export(model: nn.Module, coords: torch.Tensor, metadata: Dict[str, int], 
                             raw_cube: np.ndarray, model_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        model.eval()
        with torch.no_grad():
            predictions_flat = model(coords)
            decoded_cube = predictions_flat.view(metadata["height"], metadata["width"], metadata["bands"]).cpu().numpy()
        
        # Serialize model weights to simulate telemetry generation
        torch.save(model.state_dict(), model_path)
        logger.info(f"Successfully generated compressed weights archive: {model_path}")
        
        raw_bytes = raw_cube.nbytes
        compressed_bytes = os.path.getsize(model_path)
        
        mse = np.mean((raw_cube - decoded_cube) ** 2)
        max_pixel = max(1.0, float(np.max(raw_cube)))
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) if mse > 0 else float('inf')
        
        metrics = {
            "raw_size_mb": raw_bytes / (1024 * 1024),
            "compressed_size_mb": compressed_bytes / (1024 * 1024),
            "compression_ratio": raw_bytes / max(1, compressed_bytes),
            "reconstruction_psnr": psnr
        }
        return decoded_cube, metrics


# ==========================================
# 5. SCIENTIFIC SPECTRAL ANGLE MAPPER (SAM)
# ==========================================
def calculate_spectral_angle_mapper(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes vector divergence angles for each spatial pixel across spectral bands.
    Highly critical aerospace validation index.
    """
    H, W, B = original.shape
    vec_orig = original.reshape(-1, B)
    vec_recon = reconstructed.reshape(-1, B)
    
    dot_product = np.sum(vec_orig * vec_recon, axis=1)
    norm_orig = np.linalg.norm(vec_orig, axis=1)
    norm_recon = np.linalg.norm(vec_recon, axis=1)
    
    denominator = norm_orig * norm_recon
    denominator[denominator == 0] = 1e-8  # Protect against zero divisions
    
    cosine_sim = np.clip(dot_product / denominator, -1.0, 1.0)
    angles_radians = np.arccos(cosine_sim)
    
    sam_map = angles_radians.reshape(H, W)
    mean_sam = float(np.mean(sam_map))
    
    return mean_sam, sam_map


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    # Target execution device diagnostic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Mapping compute nodes to hardware target: {device}")
    
    WEIGHTS_FILE = "nir_compressed_weights.pth"
    
    # Controlled size parameters to keep local CPU loops execution within seconds
    H, W, B = 32, 32, 16
    logger.info(f"Generating synthetic target data dimensions: [{H}, {W}, {B}]")
    
    x, y, z = np.linspace(-2, 2, H)[:, None, None], np.linspace(-2, 2, W)[None, :, None], np.linspace(-2, 2, B)[None, None, :]
    synthetic_cube = np.sin(x * y) * np.cos(z) + np.exp(-(x**2 + y**2))
    synthetic_cube = (synthetic_cube - synthetic_cube.min()) / (synthetic_cube.max() - synthetic_cube.min())

    # Pre-process, Model Creation, Optimization
    coords, targets, metadata = HyperspectralPreprocessor.generate_training_data(synthetic_cube, device)
    nir_model = SIRENNIRRegressor(in_features=3, out_features=1, hidden_features=128, hidden_layers=4).to(device)
    
    # Train
    _ = train_nir_pipeline(nir_model, coords, targets, epochs=50, lr=1e-3, batch_size=8192)
    
    # Decompress / Evaluate Footprint
    decoded_cube, metrics = EvaluationDecompressionSuite.evaluate_and_export(
        nir_model, coords, metadata, synthetic_cube, WEIGHTS_FILE
    )
    
    # Evaluate Spectral Vector Distortion using SAM
    mean_sam_rad, sam_spatial_map = calculate_spectral_angle_mapper(synthetic_cube, decoded_cube)
    mean_sam_deg = np.degrees(mean_sam_rad)
    
    # Render final technical log metrics to standard terminal
    print("\n" + "="*50)
    print("      AEROSPACE COMPRESSION TELEMETRY MATRIX")
    print("="*50)
    print(f"Original Input Array Footprint: {metrics['raw_size_mb']:.4f} MB")
    print(f"Serialized Weights Telemetry : {metrics['compressed_size_mb']:.4f} MB")
    print(f"Operational Scaling Ratio    : {metrics['compression_ratio']:.2f}x")
    print(f"Reconstruction Accuracy PSNR : {metrics['reconstruction_psnr']:.2f} dB")
    print(f"Mean Spectral Divergence (SAM): {mean_sam_rad:.4f} rad ({mean_sam_deg:.2f}°)")
    print(f"Downstream GIS Integrity Check: {'PASSED' if mean_sam_rad < 0.1 else 'FAILED'}")
    print("="*50 + "\n")

    # Render multi-panel diagnostic verification figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(synthetic_cube[:, :, B // 2], cmap='viridis')
    axes[0].set_title(f"Original Ground Matrix (Band {B//2})")
    
    axes[1].imshow(decoded_cube[:, :, B // 2], cmap='viridis')
    axes[1].set_title(f"NIR Decompressed Matrix (Band {B//2})")
    
    im2 = axes[2].imshow(sam_spatial_map, cmap='magma')
    axes[2].set_title("Vector Divergence (SAM Map)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
