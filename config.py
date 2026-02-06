
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AlphaGenomeConfig:
    # Input
    input_length: int = 1_048_576  # 1 Mb
    num_channels: int = 4  # ACGT

    # Model Dimensions
    stem_channels: int = 64
    
    # Encoder
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 384, 512, 768, 896)
    encoder_kernels: Tuple[int, ...] = (11, 11, 11, 7, 7, 7, 7) # Example kernel sizes
    pool_kernels: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2) # Downsampling factors
    
    # Transformer / Bottleneck
    transformer_dim: int = 896
    transformer_depth: int = 8
    transformer_heads: int = 8
    transformer_window_size: int = 512 # For efficient attention
    
    # Decoder
    decoder_channels: Tuple[int, ...] = (512, 256, 128)
    # Upsampling matches the last few encoder stages in reverse or as needed for target resolution
    
    # Outputs
    num_tracks_human: int = 5930
    num_tracks_mouse: int = 1128
    
    output_resolution_1d: int = 128 # bp
    output_resolution_2d: int = 2048 # bp

    # Training
    dropout: float = 0.1
