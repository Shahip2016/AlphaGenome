
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
            
        # Transformer Bottleneck
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(current_channels, config.transformer_heads, dropout=config.dropout)
            for _ in range(config.transformer_depth)
        ])
        
        # Decoder (Upsampling)
        # Note: Decoder architecture is typically symmetric or lighter. 
        # Here we implement a simple upsampling path.
        self.decoder_stages = nn.ModuleList()
        decoder_in_channels = current_channels
        
        for out_channels in config.decoder_channels:
            stage = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBlock(decoder_in_channels, out_channels, kernel_size=5),
                ResidualBlock(out_channels, kernel_size=5)
            )
            self.decoder_stages.append(stage)
            decoder_in_channels = out_channels
            
        # Heads
        # 1D Head (Track prediction)
        self.head_1d = nn.Sequential(
            ConvBlock(decoder_in_channels, decoder_in_channels, kernel_size=1),
            nn.Conv1d(decoder_in_channels, config.num_tracks_human, kernel_size=1)
        )
        
        # 2D Head (Contact maps) - simplified projection
        self.head_2d_proj = nn.Linear(decoder_in_channels, 64) 
        self.head_2d_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 4, Length]
        x = self.stem(x)
        
        # Encoder
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)
            
        # Transformer
        for block in self.transformer_blocks:
            x = block(x)
            
        # Decoder
        for stage in self.decoder_stages:
            x = stage(x)
            
        # 1D Output
        out_1d = self.head_1d(x)
        
        # 2D Output (Contact Map)
        x_perm = x.permute(0, 2, 1) # [B, L', C]
        x_proj = self.head_2d_proj(x_perm) # [B, L', 64]
        
        # Create pairwise representation (simplified)
        out_2d_feature = x_proj.unsqueeze(2) + x_proj.unsqueeze(1) # [B, L, L, 64]
        out_2d_feature = out_2d_feature.permute(0, 3, 1, 2) # [B, 64, L, L]
        
        out_2d = self.head_2d_conv(out_2d_feature) # [B, 1, L, L]
        
        return out_1d, out_2d
