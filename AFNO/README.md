# Adaptive Fourier Neural Operator (AFNO) - 3D Implementation


The file `afno3D.py` contains a PyTorch implementation of the **Adaptive Fourier Neural Operator (AFNO)**, adapted and extended for **3D data** (i.e. inputs of shape `[B, C, D, H, W]`). The original AFNO was proposed in [*“Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers”*](https://arxiv.org/abs/2111.13587) and implemented by NVlabs in their [AFNO-transformer repository](https://github.com/NVlabs/AFNO-transformer). 

Here, the model is extended to 3D domains by modifying the architecture and FFT operations to support spectral convolutions over three spatial dimensions.
AFNO is a spectral operator designed for high-dimensional learning tasks such as fluid dynamics, turbulence modeling, and climate simulation.  
It extends Fourier Neural Operators (FNO) with adaptive spectral filtering and MLP-based non-linear mixing.

---

## Features
- **3D Support**: Works with 3D data `[B, C, D, H, W]`.
- **Spectral Convolutions**: Uses FFT-based spectral domain representation with adaptive filtering.
- **MLP Layers**: Fully-connected layers for non-linear feature transformation.
- **Patch Embedding**: Converts 3D patches into embeddings for transformer-like processing.
- **Residual Connections**: Supports double skip-connections for improved stability.
- **Configurable**: Adjustable number of blocks, hidden dimensions, sparsity thresholds, and patch sizes.

---

## Model Components
### `AFNOMlp`
A standard **MLP block** with two fully-connected layers, dropout, and activation function (default: GELU).

**Parameters:**
- `in_features` (int): Input dimension
- `latent_features` (int): Hidden dimension
- `out_features` (int): Output dimension
- `activation_fn` (nn.Module): Activation function (default: `nn.GELU`)
- `drop` (float): Dropout rate (default: 0.0)

---

### `AFNO3DLayer`
The **spectral convolutional layer**.  
Performs FFT on 3D input, applies adaptive block-wise filtering in the spectral domain, and inverse FFT back to spatial domain.

**Parameters:**
- `hidden_size` (int): Input embedding dimension
- `num_blocks` (int): Number of block diagonal weight matrices (default: 8)
- `sparsity_threshold` (float): Soft shrinkage threshold (default: 0.01)
- `hard_thresholding_fraction` (float): Fraction of modes kept (default: 1.0)
- `hidden_size_factor` (int): Expansion factor for hidden size (default: 1)

---

### `Block`
AFNO block combining:
1. **Spectral convolution (`AFNO3DLayer`)**
2. **MLP (`AFNOMlp`)**
3. Normalization + Residual Connections

**Parameters:**
- `embed_dim` (int): Embedding dimension
- `num_blocks` (int): Spectral blocks (default: 8)
- `mlp_ratio` (float): Expansion ratio in MLP (default: 4.0)
- `drop` (float): Dropout rate (default: 0.0)
- `activation_fn` (nn.Module): Default: `nn.GELU`
- `norm_layer` (nn.Module): Default: `nn.LayerNorm`
- `double_skip` (bool): Use double residual (default: True)
- `sparsity_threshold` (float): Soft shrinkage threshold (default: 0.01)
- `hard_thresholding_fraction` (float): Fraction of modes kept (default: 1.0)

---

### `PatchEmbed`
3D patch embedding layer that converts `[B, C, D, H, W]` input into patch-level embeddings.

**Parameters:**
- `inp_shape` (List[int]): Input dimensions `[D, H, W]`
- `in_channels` (int): Number of input channels
- `patch_size` (List[int]): Size of patches (default: `[1, 1, 1]`)
- `embed_dim` (int): Embedding dimension (default: 256)

---

### `AFNO`
The main **Adaptive Fourier Neural Operator** model for 3D data.

**Parameters:**
- `inp_shape` (List[int]): Input shape `[D, H, W]` (default: `[38, 38, 38]`)
- `in_channels` (int): Input channels (default: 3)
- `out_channels` (int): Output channels (default: 6)
- `patch_size` (List[int]): Patch size (default: `[1, 1, 1]`)
- `embed_dim` (int): Embedding dimension (default: 256)
- `depth` (int): Number of AFNO layers (default: 4)
- `mlp_ratio` (float): MLP expansion ratio (default: 4.0)
- `drop_rate` (float): Dropout (default: 0.0)
- `num_blocks` (int): Number of spectral blocks (default: 16)
- `sparsity_threshold` (float): Softshrink threshold (default: 0.01)
- `hard_thresholding_fraction` (float): Fraction of kept modes (default: 1.0)

---

### `AFNO-3D Pipeline`
1. **Input preparation**  
   - Input: `(B, C, D, H, W)`  

2. **Patch Embedding**  
   - Flatten 3D patches into tokens.  
   - Embed patches into latent space.  

3. **Adaptive Spectral Convolution (Core AFNO Layer)**  
   - Apply **3D FFT** to transform into Fourier space.  
   - Apply adaptive block-diagonal filtering with learnable parameters.  
   - Apply **3D IFFT** to return to spatial domain.  

4. **Residual Connections + MLP**  
   - Add normalization layers.  
   - Feed-forward block with nonlinearity.  
   - Residual skip connections.  

5. **Prediction Head**  
   - Maps back to output dimension (e.g., Reynolds stresses).  

---

## Usage

```python
import torch
from afno import AFNO

# Initialize model
model = AFNO(
    inp_shape=[38, 38, 38],
    in_channels=3,
    out_channels=6,
    patch_size=(2, 2, 2),
    embed_dim=256,
    depth=4,
    num_blocks=16,
)

# Example input (Batch=2, Channels=3, Depth=38, Height=38, Width=38)
x = torch.randn(2, 3, 38, 38, 38)

# Forward pass
out = model(x)
print(out.shape)  # torch.Size([2, 6, 38, 38, 38])

# Saving as torchscript
model_scripted = torch.jit.script(model)
model_scripted.save("afno3D.pt")

# Loading scripted model
model = torch.jit.load("afno3D.pt")
