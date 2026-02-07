
import torch
import torch.nn as nn
from config import AlphaGenomeConfig
from layers import ConvBlock, ResidualBlock, TransformerBlock

class AlphaGenome(nn.Module):
    """
    AlphaGenome: A deep learning model for genomic sequence analysis.
    Combines convolutional layers for local pattern extraction with 
    Transformers for long-range dependency modeling.
    """
    def __init__(self, config: AlphaGenomeConfig):
        super().__init__()
        self.config = config
        
        # Stem: Initial local context extraction
        self.stem = ConvBlock(
            config.num_channels, 
            config.stem_channels, 
            kernel_size=config.stem_kernel_size
        )
        
        # Encoder: Hierarchical feature extraction with downsampling
        self.encoder_stages = nn.ModuleList()
        current_channels = config.stem_channels
        
        for out_channels, kernel_size, stride in zip(
            config.encoder_channels, 
            config.encoder_kernels, 
            config.pool_kernels
        ):
            stage = nn.Sequential(
                ConvBlock(current_channels, out_channels, kernel_size=kernel_size, stride=stride),
                ResidualBlock(out_channels, kernel_size=kernel_size)
            )
            self.encoder_stages.append(stage)
            current_channels = out_channels
            
        # Transformer Bottleneck: Modeling long-range interactions
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                current_channels, 
                config.transformer_heads, 
                window_size=config.transformer_window_size,
                dropout=config.dropout
            )
            for _ in range(config.transformer_depth)
        ])
        
        # Decoder: Upsampling to target resolution
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
            
        # Task-Specific Heads
        self.head_1d = self._build_1d_head(decoder_in_channels, config.num_tracks_human)
        self.head_2d = self._build_2d_head(decoder_in_channels, config.head_2d_dim)

    def _build_1d_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """Builds the 1D track prediction head."""
        return nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def _build_2d_head(self, in_channels: int, proj_dim: int) -> nn.Module:
        """Builds the 2D contact map prediction head."""
        return nn.ModuleDict({
            'proj': nn.Linear(in_channels, proj_dim),
            'conv': nn.Conv2d(1, 1, kernel_size=1) # Simplified: operates on pairwise interaction
        })

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of AlphaGenome.
        Args:
            x: Genomic sequence tensor [Batch, 4, Length]
        Returns:
            out_1d: Track predictions [Batch, num_tracks, Length']
            out_2d: Contact maps [Batch, 1, Length'', Length'']
        """
        # Feature Extraction
        x = self.stem(x)
        
        for stage in self.encoder_stages:
            x = stage(x)
            
        for block in self.transformer_blocks:
            x = block(x)
            
        for stage in self.decoder_stages:
            x = stage(x)
            
        # 1D Prediction
        out_1d = self.head_1d(x)
        
        # 2D Prediction (Symmetric interaction map)
        # [B, C, L] -> [B, L, C]
        x_perm = x.permute(0, 2, 1)
        x_proj = self.head_2d['proj'](x_perm) # [B, L, proj_dim]
        
        # Pairwise interaction (e.g., outer sum for distance proxy)
        # [B, L, 1, proj_dim] + [B, 1, L, proj_dim] -> [B, L, L, proj_dim]
        # Then reduce to [B, 1, L, L] for the final convolution
        interaction = (x_proj.unsqueeze(2) + x_proj.unsqueeze(1)).mean(dim=-1, keepdim=True)
        interaction = interaction.permute(0, 3, 1, 2) # [B, 1, L, L]
        
        out_2d = self.head_2d['conv'](interaction)
        
        return out_1d, out_2d

