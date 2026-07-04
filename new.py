import time
import numpy as np
import matplotlib.pyplot as plt
# Import the components from your master script
from master_script import SIRENNIRRegressor, HyperspectralPreprocessor, train_nir_pipeline, EvaluationDecompressionSuite, calculate_spectral_angle_mapper, torch

# Set up testing variables
hidden_feature_options = [32, 64, 128, 256]
results = []

# Generate larger synthetic cube to show real compression gains
H, W, B = 64, 64, 32 
x, y, z = np.linspace(-2, 2, H)[:, None, None], np.linspace(-2, 2, W)[None, :, None], np.linspace(-2, 2, B)[None, None, :]
synthetic_cube = np.sin(x * y) * np.cos(z)
synthetic_cube = (synthetic_cube - synthetic_cube.min()) / (synthetic_cube.max() - synthetic_cube.min())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coords, targets, metadata = HyperspectralPreprocessor.generate_training_data(synthetic_cube, device)

print("Starting Pareto Frontier Sweep...")
for features in hidden_feature_options:
    print(f"\nTesting hidden_features = {features}")
    
    # Initialize model with current variable width
    model = SIRENNIRRegressor(in_features=3, out_features=1, hidden_features=features, hidden_layers=4).to(device)
    
    # Train quickly
    train_nir_pipeline(model, coords, targets, epochs=30, lr=1e-3, batch_size=8192)
    
    # Evaluate
    decoded_cube, metrics = EvaluationDecompressionSuite.evaluate_and_export(
        model, coords, metadata, synthetic_cube, f"temp_weights_{features}.pth"
    )
    mean_sam, _ = calculate_spectral_angle_mapper(synthetic_cube, decoded_cube)
    
    results.append({
        "features": features,
        "ratio": metrics["compression_ratio"],
        "psnr": metrics["reconstruction_psnr"],
        "sam": mean_sam
    })

# ==========================================
# PLOT THE PARETO FRONTIER
# ==========================================
ratios = [r["ratio"] for r in results]
psnrs = [r["psnr"] for r in results]
labels = [f"{r['features']} hidden features" for r in results]

plt.figure(figsize=(8, 6))
plt.scatter(ratios, psnrs, color='red', s=100, zorder=3)

for i, txt in enumerate(labels):
    plt.annotate(txt, (ratios[i], psnrs[i]), xytext=(5, 5), textcoords='offset points')

plt.plot(ratios, psnrs, linestyle='--', color='blue', alpha=0.5)
plt.title("NIR Space Compression Pareto Frontier")
plt.xlabel("Compression Ratio (Higher is Better)")
plt.ylabel("Reconstruction PSNR in dB (Higher is Better)")
plt.grid(True, linestyle=':')
plt.show()
