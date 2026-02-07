
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    """Standard convolution block with batch norm and activation."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        dilation: int = 1, 
        activation: str = 'gelu'
    ):
        super().__init__()
        # Use automated 'same' padding for stride 1 if possible, else manual calculation
        if stride == 1:
            padding = (kernel_size - 1) // 2 * dilation
        else:
            padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU() if activation == 'gelu' else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size, dilation=dilation, activation='gelu'),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same', bias=False),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=512, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [Batch, Channels, Length] -> [Batch, Length, Channels] for Attention
        x_perm = x.permute(0, 2, 1)
        
        # Simple windowed attention approximation or full attention if length permits (simplified here)
        # In a real 1Mb scenario, we would use windowed attention or efficient transformers.
        # Here we assume the input to transformer is already downsampled enough or we operate on chunks.
        
        # Pre-norm architecture
        x_norm = self.norm1(x_perm)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_perm = x_perm + self.dropout(attn_out)
        
        x_norm = self.norm2(x_perm)
        mlp_out = self.mlp(x_norm)
        x_perm = x_perm + mlp_out
        
        return x_perm.permute(0, 2, 1) # Back to [Batch, Channels, Length]
