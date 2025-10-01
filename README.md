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

## File Structure

| Model | File |
|-------|------|
| **AFNO** | `AFNO/afno.py` |
| **FNO** | `FNO/fno.py` |
| **ResNet18_3D** | `ResNet3D/resnet3d.py` |

---


