# Data Module

This module handles all data-related operations for the Spiking Neural Network (SNN) project.

## Overview

The data module is responsible for:
- Loading and preprocessing raw data
- Converting continuous data into spike trains
- Managing data batching and iteration
- Implementing various spike encoding strategies

## Components

### 1. DataLoader (`data_loader.py`)
Handles the loading and batching of datasets:
- `SNNDataset`: Custom PyTorch Dataset class for handling SNN data
- `SNNDataLoader`: Wrapper class that provides train/validation data loaders

### 2. Preprocessor (`preprocessor.py`)
Implements data preprocessing and spike encoding methods:
- Rate coding: Converts continuous values to probabilistic spike trains
- Temporal coding: Implements time-based spike encoding
- Extensible design for additional encoding methods

## Usage Example

```python
from data.preprocessor import Preprocessor
from data.data_loader import SNNDataLoader

# Initialize preprocessor with desired encoding method
preprocessor = Preprocessor(encoding_method='rate', time_steps=100)

# Configure data loading
dataset_params = {
    'train_path': 'path/to/train/data',
    'val_path': 'path/to/val/data',
    'batch_size': 32
}

# Create data loaders
data_loader = SNNDataLoader(dataset_params)
train_loader, val_loader = data_loader.get_data_loaders()