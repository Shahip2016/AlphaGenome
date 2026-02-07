
import torch
import time
import logging
from config import AlphaGenomeConfig
from model import AlphaGenome

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_model_verification():
    """Verifies the model architecture and forward pass."""
    logger.info("Initializing AlphaGenome with test configuration...")
    
    # Reduced config for faster testing
    config = AlphaGenomeConfig(
        input_length=65536,
        num_channels=4,
        stem_channels=16,
        encoder_channels=(32, 64),
        encoder_kernels=(5, 5),
        pool_kernels=(2, 2),
        transformer_dim=64,
        transformer_depth=1,
        transformer_heads=2,
        decoder_channels=(32,),
        num_tracks_human=10
    )
    
    try:
        model = AlphaGenome(config)
        logger.info("Model successfully initialized.")
        
        # [Batch, Channels, Length]
        batch_size = 1
        x = torch.randn(batch_size, config.num_channels, config.input_length)
        logger.info(f"Input tensor shape: {x.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            out_1d, out_2d = model(x)
        duration = time.time() - start_time
        
        logger.info(f"Forward pass completed in {duration:.4f}s")
        logger.info(f"Output 1D shape: {out_1d.shape}")
        logger.info(f"Output 2D shape: {out_2d.shape}")
        
        # Verification of shapes
        # Enc: /4 -> 16384
        # Dec: *2 -> 32768
        expected_1d = (batch_size, config.num_tracks_human, 32768)
        expected_2d = (batch_size, 1, 32768, 32768)
        
        assert out_1d.shape == expected_1d, f"1D shape mismatch: {out_1d.shape} != {expected_1d}"
        assert out_2d.shape == expected_2d, f"2D shape mismatch: {out_2d.shape} != {expected_2d}"
        
        logger.info("Aesthetic and structural verification: PASSED")
        
    except Exception as e:
        logger.error(f"Verification FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    run_model_verification()

