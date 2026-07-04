# Spaceborne Implicit Compression & Real-Time Telemetry Pipeline

An advanced Aerospace Software Engineering project utilizing **Neural Implicit Representations (NIR)** to compress high-dimensional satellite hyperspectral data cubes into continuous coordinate-based neural fields. 

Built using PyTorch, this pipeline leverages **SIREN (Sinusoidal Representation Networks)** architectures to enable arbitrary-resolution spatial-spectral decoding, optimized for extreme spaceflight compute and thermal constraints.

---

## 🛰️ Project Overview & Problem Statement
Traditional spaceborne hyperspectral sensors capture hundreds of narrow, contiguous spectral bands, generating massive data footprints. Transmitting these huge raw data cubes across constrained satellite downlink channels introduces a severe telemetry bottleneck. Traditional block-compression algorithms (like JPEG2000 or 3D-GZIP) require high onboard memory allocation and cause blocky artifacts.

**The Solution:** This framework compresses a massive 3D hyperspectral cube directly into the weight parameters of a coordinate regressor neural network ($f_\theta: (x, y, \lambda) \to I$). 

Instead of saving a massive pixel matrix, we save the compact model weights (`.pth`), mapping continuous spatial-spectral positions to reflectance intensities.

---

## 🛠️ Architecture & Key Innovations

### 1. Sinusoidal Activation Fields (SIREN)
Standard ReLU networks suffer from "spectral bias" and cannot accurately reconstruct the fine high-frequency edge gradients inherent in multi-band remote sensing imagery. This framework implements a multi-layer periodic activation function:
$$\text{SineActivation}(x) = \sin(\omega_0 \cdot x)$$
This allows the network's derivatives to remain stable and clean, preserving intricate physical spatial patterns and sharp absorption lines across the spectral spectrum.

### 2. Thermal-Aware Adaptive Optimization Engine
To prevent onboard hardware damage from core-pinning and volatile CPU cache power consumption, the pipeline integrates defensive execution guardrails:
* **Micro-Throttling:** Tiny execution pauses ($2\text{ms}$) built directly into the batch loops allow consistent core heat dissipation.
* **Dynamic Hyperparameter Scaling:** The pipeline dynamically matches batch constraints and learning rates to network dimensions—preventing convergence collapse while maximizing learning stability.
* **Layout-Agnostic Preprocessing:** Uses memory-layout resilient tensors (`.reshape()`) to handle arbitrary scientific matrix strides safely.

---

## 📊 Experimental Results & Pareto Frontiers

The system was evaluated across multiple network configurations (`Width: 32` to `256`) to map out data compression trade-offs on real orbital imagery (NASA's AVIRIS Indian Pines flight payload).

### Performance Metrics Table
| Profile Architecture | Compression Ratio | Reconstruction Fidelity (PSNR) | Operational Profile |
| :--- | :---: | :---: | :--- |
| **SIREN-Width-32** | **~19.7x** | ~37.1 dB | Low-power / Fast-downlink telemetry |
| **SIREN-Width-64** | **~6.1x** | ~38.4 dB | Balanced routine orbital monitoring |
| **SIREN-Width-128** | **~1.6x** | ~53.0 dB | Standard high-fidelity data preservation |
| **SIREN-Width-256** | **~0.4x** | **~74.2 dB** | Extreme precision scientific validation |

### The Pareto Frontier Trade-Off
The network successfully generates a textbook mathematical **Pareto Frontier**, mapping a predictable trade-off between file footprint reduction and image quality. This allows ground stations to dynamically adjust compression profiles depending on varying satellite power states or downlink windows.

---

## 🖥️ Live Real-Time Ground Station Simulation
The project features a native desktop telemetry simulator. Because the trained neural network acts as a continuous function, it does not need to unpack a massive file array to read data. The ground station can query individual spatial coordinates on-the-fly, achieving real-time data streaming:

* **Decoding Performance Delay:** ~69.0ms per hyperspectral band.
* **Visualization Layer:** Sequential streaming across the shortwave and near-infrared spectral spectrum.


![Pareto Frontier](83d3aa.png)
![Live Telemetry Stream](8f2dc9.png)
