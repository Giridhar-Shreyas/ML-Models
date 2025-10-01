# Fourier Neural Operator (FNO) - 3D Implementation

The file `fno.py`contains a PyTorch implementation of the **Fourier Neural Operator (FNO)**, adapted and extended for **3D data** (i.e. inputs of shape `[B, C, D, H, W]`).  
The original FNO was proposed in [*“Fourier Neural Operator for Parametric Partial Differential Equations”*](https://arxiv.org/abs/2010.08895) and implemented by [Zongyi Li et al.](https://github.com/ixScience/fourier_neural_operator/tree/master).

Here, the model is extended to 3D domains by modifying the architecture and FFT operations to support spectral convolutions over three spatial dimensions.  
FNO is designed for learning operators on functions, and has shown strong performance in solving PDEs, fluid dynamics, turbulence modeling, and climate simulations.

---

## Features
- **3D Support**: Works with 3D data `[B, C, D, H, W]`.
- **Spectral Convolutions**: Uses 3D FFT-based convolutions with learnable complex weights in Fourier space.
- **MLP Layers**: Pointwise 1×1×1 convolutions for non-linear feature transformation.
- **Residual Connections**: Parallel convolutional skip paths (`W` layers) for stability.
- **Configurable**: Adjustable number of Fourier modes, hidden width, and depth.
- **Zero-shot-super-resolution**: Train on coarser grids and infer on finer grids without needing to retrain.


---

## Model Components

### `SpectralConv3d`
A **3D Fourier convolution layer**.  
Performs forward FFT, multiplies a limited number of Fourier modes by learnable complex weights, and applies inverse FFT to return to spatial domain.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `modes1` (int): Number of Fourier modes kept along depth
- `modes2` (int): Number of Fourier modes kept along height
- `modes3` (int): Number of Fourier modes kept along width

---

### `MLP`
A lightweight **pointwise MLP block** implemented as two `Conv3d` layers with GELU activation.

**Parameters:**
- `in_channels` (int): Input channels
- `out_channels` (int): Output channels
- `mid_channels` (int): Hidden dimension

---

### `FNO`
The main **Fourier Neural Operator** model for 3D data.

**Parameters:**
- `modes1, modes2, modes3` (int): Number of Fourier modes retained along each spatial axis (x,y,z respectively) (default 19).
- `width` (int): Feature dimensionality of the embedding space (default 128).
- `padding` (int): Padding applied to handle non-periodic domains (default: 6).
- `in_channels` (int): Input channels (default: 3).
- `out_channels` (int): Output channels (default: 6).


---


### `FNO Pipeline`
1. **Input preparation**  
    - Input: `(B, C, D, H, W)`  
     - `B`: Batch size  
     - `C`: Number of input channels (e.g., velocity components)  
     - `D, H, W`: Depth, Height, Width of the 3D grid  

2. **Lifting with Linear Layer**
   - Apply point wise linear layer `p` to map input channels `C` → `width` (higher dimensional embedding).

3. **Spectral Convolution**  
   - Apply **3D FFT** to transform data into frequency space.  
   - Truncate high-frequency modes.  
   - Apply learnable linear weights in Fourier domain.  
   - Apply **3D IFFT** to return to spatial domain.  

4. **Residual Connections + Nonlinearities**  
   - MLP layer for nonliner channel mixing
   - Combine spectral features with skip connections.  
   - Apply pointwise linear layers + GELU/ReLU.  

5. **Prediction Head**  
   - Output shape: `(B, C_out, D, H, W)`  

---

## Usage

```python
import torch

# Initialize model
model = FNO(
    modes1 = 19, 
    modes2 = 19, 
    modes3 = 19, 
    width=128, 
    in_channels=3, 
    out_channels=6
)

# Example input (Batch=2, Channels=3, Depth=38, Height=38, Width=38)
x = torch.randn(2, 3, 38, 38, 38)

# Forward pass
out = model(x)
print(out.shape)  # torch.Size([2, 6, 38, 38, 38])

# Saving as torchscript
model_scripted = torch.jit.script(model)
model_scripted.save("fno3D.pt")

# Loading scripted model
model = torch.jit.load("fno3D.pt")
