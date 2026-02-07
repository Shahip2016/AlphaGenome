
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AlphaGenomeConfig:
    """Configuration for the AlphaGenome model."""
    
    # Input Processing
    input_length: int = 1_048_576  # 1 Mb genomic sequence
    num_channels: int = 4          # DNA bases: A, C, G, T
    
    # Model Architecture: Stem
    stem_channels: int = 64
    stem_kernel_size: int = 15
    
    # Model Architecture: Encoder (Downsampling)
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 384, 512, 768, 896)
    encoder_kernels: Tuple[int, ...] = (11, 11, 11, 7, 7, 7, 7)
    pool_kernels: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2)
    
    # Model Architecture: Transformer Bottleneck
    transformer_dim: int = 896
    transformer_depth: int = 8
    transformer_heads: int = 8
    transformer_window_size: int = 512  # For efficient local attention
    
    # Model Architecture: Decoder (Upsampling)
    decoder_channels: Tuple[int, ...] = (512, 256, 128)
    
    # Output Heads
    num_tracks_human: int = 5930
    num_tracks_mouse: int = 1128
    head_2d_dim: int = 64
    
    # Resolutions (base pairs)
    output_resolution_1d: int = 128
    output_resolution_2d: int = 2048

    # Training Hyperparameters
    dropout: float = 0.1

