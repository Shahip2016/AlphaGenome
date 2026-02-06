# AlphaGenome: Unified DNA Sequence Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

**AlphaGenome** is a state-of-the-art deep learning model designed to predict functional genomic measurements from DNA sequences. By processing up to **1 Mb** of DNA context at base-pair resolution, it captures long-range regulatory interactions and predicts thousands of genomic tracks across diverse modalities.

---

## 🚀 Key Features

*   **Long-Range Context**: Processes **1,048,576 bp** (1 Mb) of input sequence to capture distal regulatory elements (enhancers, promoters).
*   **Multimodal Prediction**: Simultaneously predicts:
    *   Gene Expression (RNA-seq, CAGE, PRO-cap)
    *   Chromatin Accessibility (DNase, ATAC-seq)
    *   Histone Modifications & TF Binding
    *   3D Chromatin Contact Maps (Hi-C)
*   **High Resolution**: Outputs predictions at variable resolutions (1bp for fine-grained tracks, 128bp for broad features).
*   **Unified Architecture**: Combines the efficiency of CNNs (U-Net backbone) with the global context of Transformers.

## 🏗️ Model Architecture

The AlphaGenome architecture leverages a U-Net style backbone with a Transformer bottleneck to efficiently process ultra-long sequences.

```mermaid
graph TD
    Input[DNA Sequence (1Mb)] --> Stem
    Stem --> Encoder
    Encoder -->|Downsampling| Transformer[Transformer Bottleneck]
    Transformer --> Decoder
    Decoder -->|Upsampling| Head1D[1D Track Heads]
    Decoder -->|Upsampling| Head2D[2D Contact Map Heads]
    Head1D --> Tracks[Genomic Tracks]
    Head2D --> Maps[Contact Maps]
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (64-bit required)
- CUDA-capable GPU (recommended for training)

### Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:Shahip2016/AlphaGenome.git
   cd AlphaGenome
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Verification
To verify the model architecture and run a forward pass with random data:

```bash
python test_model.py
```

*Note: The test script uses reduced dimensions to allow execution on standard hardware.*

### Configuration
Model hyperparameters are defined in `config.py`. You can adjust:
*   `input_length`: Sequence length (default: 2^20).
*   `encoder_channels`: Network depth and width.
*   `num_tracks_human`: Number of output targets.

## 📂 Project Structure

```
AlphaGenome/
├── config.py         # Configuration dataclasses
├── layers.py         # Core neural network building blocks (Conv, attn)
├── model.py          # Full AlphaGenome architecture assembly
├── test_model.py     # Verification script
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## 📜 Citation

If you use this implementation, please cite the original AlphaGenome paper:

> Avsec, Ž. et al. "Advancing regulatory variant effect prediction with AlphaGenome." *Nature* 649 (2026).

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
