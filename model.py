
import torch
import torch.nn as nn
from config import AlphaGenomeConfig
from layers import ConvBlock, ResidualBlock, TransformerBlock

class AlphaGenome(nn.Module):
    def __init__(self, config: AlphaGenomeConfig):
        super().__init__()
        self.config = config
        
        # Stem
        self.stem = ConvBlock(config.num_channels, config.stem_channels, kernel_size=15)
        
        # Encoder (Downsampling)
        self.encoder_stages = nn.ModuleList()
        current_channels = config.stem_channels
        
        for out_channels, kernel_size, stride in zip(config.encoder_channels, config.encoder_kernels, config.pool_kernels):
            stage = nn.Sequential(
                ConvBlock(current_channels, out_channels, kernel_size=kernel_size, stride=stride),
                ResidualBlock(out_channels, kernel_size=kernel_size)
            )
            self.encoder_stages.append(stage)
            current_channels = out_channels

    def forward(self, x):
        # x: [Batch, 4, Length]
        x = self.stem(x)
        
        # Encoder
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)
        
        return x
