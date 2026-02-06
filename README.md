# AlphaGenome Implementation

PyTorch implementation of the AlphaGenome model for predicting functional genomic measurements from DNA sequences.

## Overview
AlphaGenome processes 1Mb of DNA sequence to predict thousands of genomic tracks (gene expression, splicing, chromatin accessibility, etc.) at base-pair resolution.

## Architecture
- **Input**: 1Mb DNA sequence (one-hot encoded).
- **Backbone**: U-Net style architecture with a Transformer bottleneck.
- **Heads**: 
  - 1D Head: Predicts genomic tracks.
  - 2D Head: Predicts chromatin contact maps.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run verification:
   ```bash
   python test_model.py
   ```
