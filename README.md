# 3D Operator and CNN Models for 3D spatial Data

This repository contains PyTorch implementations of three neural architectures for **3D spatial data** `[B, C, D, H, W]`:

1. **Adaptive Fourier Neural Operator (AFNO-3D)**  
    - **Description:** Adaptive Fourier Neural Operator extended to 3D with FFT-based spectral convolutions and MLP-based nonlinear mixing.  
2. **Fourier Neural Operator (FNO-3D)**  
    - **Description:** Fourier Neural Operator extended to 3D; learns mappings in Fourier space using a fixed number of modes. Supports **zero-shot super-resolution**.  
3. **ResNet-18 extended to 3D (ResNet18_3D)** 
    - **Description:** Standard ResNet-18 architecture extended to 3D using `Conv3d`, `BatchNorm3d`, and `ConvTranspose3d`. Suitable for 3D spatial regression and 3D segmentation.  

These models are designed for tasks such as fluid dynamics, turbulence modeling, PDE learning, and 3D spatial regression.

---

## Features & Comparison

| Feature | AFNO-3D | FNO-3D | ResNet18_3D |
|---------|---------|--------|-------------|
| **Input type** | `[B, C, D, H, W]` | `[B, C, D, H, W]` | `[B, C, D, H, W]` |
| **Architecture type** | Spectral + MLP | Spectral | CNN (Residual) |
| **Spectral Convolutions** | ✅ Adaptive FFT/IFFT with block-diagonal filtering | ✅ FFT/IFFT with fixed Fourier modes | ❌ |
| **MLP Layers** | ✅ Nonlinear mixing post-FFT | ✅ Pointwise 1×1×1 Conv | ✅ Fully convolutional (ReLU activations) |
| **Residual Connections** | ✅ Double skip connections in blocks | ✅ Skip paths with pointwise projections | ✅ Classic ResNet skips via `BasicBlock` |
| **Patch Embedding** | ✅ Converts 3D patches to latent embeddings | ❌ Works directly on full grid | ❌ Works directly on full grid |
| **Zero-Shot Super Resolution** | ❌ | ✅ Can train on coarse grids, infer on finer grids | ❌ |
| **Typical Use Cases** | Fluid dynamics, PDEs, turbulence, transformer-like processing | PDE learning, super-resolution, operator learning | 3D image segmentation, 3D spatial regression, general CNN tasks |

---

## Data Loading

### `load_sclice_local(filename, domain_size)`
Loads and reshapes CSV data into 3D NumPy arrays of size `domain_size × domain_size × domain_size` for velocity (`UUX`, `UUY`, `UUZ`) and Reynolds stress tensor (`TAU_xx`, `TAU_yy`, etc.) components.

**Parameters:**  
- `filename` *(str)* — Path to the input CSV file containing the simulation data.  
- `domain_size` *(int)* — Size of each spatial dimension of the cubic domain.

**Returns:**  
- `true_taus` — Ground truth stress tensor components.  
- `pred_taus` — Model-inferred stress tensor components.  
- `uumeans` — Mean velocity components (`UUMEAN_x`, `UUMEAN_y`, `UUMEAN_z`).  
- `uus` — Instantaneous velocity components (`UUX`, `UUY`, `UUZ`).  

---

## Visualization

### `plot_uumean_vs_uu()`
Compares mean and instantaneous velocity fields.

**Parameters:**  
- `uumeans` *(np.ndarray)* — Mean velocity field components.  
- `uus` *(np.ndarray)* — Instantaneous velocity field components.  
- `save` *(bool)* — Whether to save the plot as an image.  
- `domain_size` *(int)* — Size of the domain in each spatial dimension.  
- `i`, `j`, `k` *(int, optional)* — Slice indices along the x, y, z dimensions.  
- `filename` *(str, optional)* — Output filename if `save=True`.  

---

### `plot_tau_vs_inferre_tau()`
Plots true vs. inferred stress tensor components.

### `plot_tau_vs_inferre_tau_l2()`
Adds L2 difference visualization between true and predicted tensors.

### `plot_tau_vs_inferre_tau_full()`
Extended comparison including MSE, SSIM, PSNR, and fractional differences.

**Parameters (for all three):**  
- `true_taus` *(np.ndarray)* — True stress tensor components.  
- `pred_taus` *(np.ndarray)* — Predicted stress tensor components.  
- `save` *(bool)* — Whether to save the plot as an image.  
- `domain_size` *(int)* — Size of the domain in each spatial dimension.  
- `title_name` *(str)* — Title for the plot.  
- `i`, `j`, `k` *(int, optional)* — Slice indices along the x, y, z dimensions.  
- `filename` *(str, optional)* — Output filename if `save=True`.  

---

## Example Usage

```python
from utils import *

# Load simulation data
true_taus, pred_taus, uumeans, uus = load_sclice_local("data.csv", domain_size=38)

# Plot predicted vs true stress tensors
plot_tau_vs_inferre_tau(
    true_taus=true_taus,
    pred_taus=pred_taus,
    save=False,
    domain_size=38,
    title_name='Stress Tensor Comparison',
    k=19,
    filename='comparison.png'
)
```
---

## File Structure

| Model | File |
|-------|------|
| **AFNO** | `AFNO/afno.py` |
| **FNO** | `FNO/fno.py` |
| **ResNet18_3D** | `ResNet3D/resnet3d.py` |
| **utiols** | `utils.py`|

---


