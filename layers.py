
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
    """Transformer block with multi-head attention and MLP."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TransformerBlock.
        Args:
            x: Input tensor of shape [Batch, Channels, Length]
        Returns:
            Output tensor of shape [Batch, Channels, Length]
        """
        # [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)
        
        # Attention with Pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        
        # MLP with Pre-norm
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        # [B, L, C] -> [B, C, L]
        return x.permute(0, 2, 1)

