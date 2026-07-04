import os
import sys
import math
import time
import logging
import urllib.request  # Fetches live telemetry payload files
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Configure logging for R&D traceability
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')
logger = logging.getLogger("NIR_Production_FlightTest")

# ==========================================
# 1. CORE ARCHITECTURE (SIREN Network)
# ==========================================
class SineActivation(nn.Module):
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SIRENNIRRegressor(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 1, 
                 hidden_features: int = 128, hidden_layers: int = 4, 
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        self.net = []
        
        first_layer = nn.Linear(in_features, hidden_features)
        self.init_siren_layer(first_layer, is_first=True, omega_0=first_omega_0)
        self.net.append(first_layer)
        self.net.append(SineActivation(omega_0=first_omega_0))
        
        for _ in range(hidden_layers):
            layer = nn.Linear(hidden_features, hidden_features)
            self.init_siren_layer(layer, is_first=False, omega_0=hidden_omega_0)
            self.net.append(layer)
            self.net.append(SineActivation(omega_0=hidden_omega_0))
            
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
        with torch.no_grad():
            num_input = layer.in_features
            bounds = 1 / num_input if is_first else math.sqrt(6 / num_input) / omega_0
            layer.weight.data.uniform_(-bounds, bounds)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class HyperspectralPreprocessor:
    @staticmethod
    def generate_training_data(data_cube: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        height, width, bands = data_cube.shape
        metadata = {"height": height, "width": width, "bands": bands}
        
        x_coords = torch.linspace(-1, 1, steps=height)
        y_coords = torch.linspace(-1, 1, steps=width)
        lambda_coords = torch.linspace(-1, 1, steps=bands)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, lambda_coords, indexing='ij')
        
        coords_flat = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)
        
        # TARGETED CORRECTION: Reshape flattens non-contiguous memory slices seamlessly
        targets_flat = torch.tensor(data_cube, dtype=torch.float32).reshape(-1, 1).to(device)
        
        return coords_flat, targets_flat, metadata


# ==========================================
# 2. THERMAL-AWARE ADAPTIVE PIPELINE ENGINE
# ==========================================
def train_nir_pipeline_thermal_safe(model: nn.Module, coords: torch.Tensor, targets: torch.Tensor, 
                                    features_width: int) -> None:
    epochs = 60 if features_width >= 128 else 40
    lr = 5e-4 if features_width >= 128 else 1e-3
    batch_size = 4096 if features_width >= 128 else 8192

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    dataset_size = coords.shape[0]
    
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i: i + batch_size]
            predictions = model(coords[indices])
            loss = criterion(predictions, targets[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        scheduler.step(avg_loss)
        time.sleep(0.002) # Core cool-down safety margin
        
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f" -> Epoch {epoch:03d}/{epochs:03d} | Current Processing Loss: {avg_loss:.6f}")


class EvaluationDecompressionSuite:
    @staticmethod
    def evaluate_and_export(model: nn.Module, coords: torch.Tensor, metadata: Dict[str, int], 
                             raw_cube: np.ndarray, model_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        model.eval()
        with torch.no_grad():
            predictions_flat = model(coords)
            decoded_cube = predictions_flat.view(metadata["height"], metadata["width"], metadata["bands"]).cpu().numpy()
        
        torch.save(model.state_dict(), model_path)
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
        if os.path.exists(model_path):
            os.remove(model_path)
            
        return decoded_cube, metrics


def run_realtime_groundstation_simulation(model: nn.Module, metadata: Dict[str, int], device: torch.device):
    logger.info("Initializing Operational Live Ground-Station Decompression View...")
    model.eval()
    H, W, B = metadata["height"], metadata["width"], metadata["bands"]
    
    # Disable interactive mode and build a dedicated clean figure canvas
    plt.ioff() 
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Initialize frame with structural dimensions
    band_frame_init = np.zeros((H, W))
    im = ax.imshow(band_frame_init, cmap='viridis', vmin=0, vmax=1)
    title = ax.set_title("Operational Downlink Stream | Band: 1")
    
    x_coords = torch.linspace(-1, 1, steps=H)
    y_coords = torch.linspace(-1, 1, steps=W)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    
    print("\n>>> Streaming Live Decompressed Telemetry Frame Data <<<\n")
    
    # We will loop through the bands using explicit background canvas flushes
    for band_idx in range(B):
        start_time = time.time()
        z_val = -1.0 + (2.0 * band_idx / (B - 1))
        grid_z = torch.full((H, W), z_val)
        
        band_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).to(device)
        
        with torch.no_grad():
            band_pred = model(band_coords)
            band_frame = band_pred.view(H, W).cpu().numpy()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update elements
        im.set_data(band_frame)
        ax.set_title(f"Live Spatial Slice | Band: {band_idx+1}/{B} | Decoding Latency: {latency_ms:.1f}ms")
        
        # FORCE render loop to draw directly onto the native OS desktop thread
        plt.draw()
        plt.pause(0.05)
        
    print("\n>>> Telemetry stream sequence complete. Presenting final static matrix plot. <<<\n")
    # Keep the window frozen open at the final band until you click the "X" close button
    plt.show()


# ==========================================
# 3. DATA INGESTION ENGINE
# ==========================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        torch.set_num_threads(2)  # Active thermal throttling prevention
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_file = "Indian_pines_corrected.mat"
    mirror_url = "https://github.com/researcher111/LearningHyperspectral/raw/master/starter-Pines/Indian_pines_corrected.mat"
    
    if not os.path.exists(data_file):
        logger.info(f"Payload target '{data_file}' absent from local disk. Requesting telemetry stream from mirror...")
        try:
            urllib.request.urlretrieve(mirror_url, data_file)
            logger.info("Telemetry download complete. Ingesting satellite data matrix...")
        except Exception as e:
            logger.error(f"Download stream interrupted: {e}. Reverting to mathematical fallback matrix.")

    if os.path.exists(data_file):
        mat_data = loadmat(data_file)
        key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        raw_cube = mat_data[key].astype(np.float32)
        
        # Crop data down safely to 50x50 spatial pixels and 24 bands for smooth calculation
        raw_cube = raw_cube[:50, :50, :24]
        raw_cube = (raw_cube - raw_cube.min()) / (raw_cube.max() - raw_cube.min())
    else:
        H, W, B = 48, 48, 24
        x, y, z = np.linspace(-2, 2, H)[:, None, None], np.linspace(-2, 2, W)[None, :, None], np.linspace(-2, 2, B)[None, None, :]
        raw_cube = np.sin(x * y) * np.cos(z) + np.exp(-(x**2 + y**2))
        raw_cube = (raw_cube - raw_cube.min()) / (raw_cube.max() - raw_cube.min())

    coords, targets, metadata = HyperspectralPreprocessor.generate_training_data(raw_cube, device)
    
    features = 128
    logger.info(f"Launching Operational Compression Run (Width: {features} hidden units)...")
    
    flight_model = SIRENNIRRegressor(in_features=3, out_features=1, hidden_features=features, hidden_layers=4).to(device)
    train_nir_pipeline_thermal_safe(flight_model, coords, targets, features)
    
    _, metrics = EvaluationDecompressionSuite.evaluate_and_export(
        flight_model, coords, metadata, raw_cube, "mission_telemetry.pth"
    )
    
    print("\n" + "="*55)
    print("      REAL-TIME MISSION TEST FLIGHT LOGS")
    print("="*55)
    print(f"Sensor Payload Target Dimensions : [{metadata['height']}x{metadata['width']}x{metadata['bands']}]")
    print(f"Achieved Compression Ratio Factor: {metrics['compression_ratio']:.2f}x")
    print(f"Telemetry Reconstruction PSNR    : {metrics['reconstruction_psnr']:.2f} dB")
    print("="*55 + "\n")
    
    # Fire off live simulation using the real ground file data
    run_realtime_groundstation_simulation(flight_model, metadata, device)
