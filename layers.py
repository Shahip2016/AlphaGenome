
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, activation='gelu'):
        super().__init__()
        if padding == 'same' and stride == 1:
            padding = (kernel_size - 1) // 2 * dilation
        elif padding == 'same':
             padding = (kernel_size - 1) // 2 # Approximation for stride > 1
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU() if activation == 'gelu' else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size, dilation=dilation, activation='gelu'),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same', bias=False),
            nn.BatchNorm1d(channels)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))
