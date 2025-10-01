# 3D ResNet-18 Implementation

The file `resnet.py` contains a PyTorch implementation of **ResNet-18 adapted for 3D data** (inputs of shape `[B, C, D, H, W]`).  
It extends the standard ResNet architecture to 3D by using `Conv3d`, `BatchNorm3d`, and `ConvTranspose3d` layers.

---

## Features
- **3D Support**: Works with data `[B, C, D, H, W]`.
- **Residual Connections**: Implements the classic ResNet skip connections via the `BasicBlock`.
- **Batch Normalization**: Stabilizes training and accelerates convergence.
- **Upsampling Layer**: Includes a `ConvTranspose3d` layer to upsample feature maps for output prediction.
- **Configurable Blocks**: `_make_layer` allows easy adjustment of number of residual blocks and channels.

---

## Model Components

### `BasicBlock`
A **residual block** with two 3×3×3 convolutions and skip connections.  

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `stride` (int): Stride for the first convolution (default 1)

---

### `ResNet18_3D`
The main **3D ResNet-18 model**.


**Parameters:**
- `in_channels` (int): Number of input channels (default 3)
- `out_channels` (int): Number of output channels (default 6)

---

### `ResNet18_3D Pipeline`

1. **Input Preparation**  
   - Input: `(B, C, D, H, W)`  
   - `B`: Batch size  
   - `C`: Number of channels  
   - `D, H, W`: Depth, Height, Width of 3D volume

2. **Initial Convolutional Layers**  
   - Conv3D + BatchNorm3D + ReLU  
   - Prepare features for residual blocks

3. **Residual Blocks**  
   - `BasicBlock` layers with skip connections  
   - Downsample spatial dimensions when `stride=2`  
   - Apply element-wise addition (residual) followed by ReLU

4. **Upsampling & Output Head**  
   - `ConvTranspose3d` doubles spatial resolution  
   - Sequential conv layers map features to output channels  

---

## Usage

```python
import torch
from resnet3d import ResNet18_3D

# Initialize model
model = ResNet18_3D(in_channels=3, out_channels=6)

# Example input (Batch=2, Channels=3, Depth=38, Height=38, Width=38)
x = torch.randn(2, 3, 38, 38, 38)

# Forward pass
out = model(x)
print(out.shape)  # torch.Size([2, 6, 38, 38, 38])

# Saving as torchscript
model_scripted = torch.jit.script(model)
model_scripted.save("resnet18_3d.pt")

# Loading scripted model
model = torch.jit.load("resnet18_3d.pt")
