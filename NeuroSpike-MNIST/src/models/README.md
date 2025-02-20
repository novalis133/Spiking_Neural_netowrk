# Models Module

This module contains the Spiking Neural Network (SNN) model implementations and architectures.

## Overview

The models module provides:
- Base SNN model architectures
- Custom layer implementations
- Model configuration utilities
- Factory methods for model creation

## Components

### 1. SNN Model (`snn_model.py`)
Core model implementations:
- Base SNN model class
- Different network architectures
- Model configuration handling

### 2. Layers (`layers.py`)
Custom layer implementations:
- Spiking neural layers
- Pooling layers
- Custom activation functions
- Dropout and normalization layers

## Usage Example

```python
from models.snn_model import SNNModel
from models.layers import SpikingLayer

# Create a simple SNN model
model = SNNModel(
    input_size=784,
    hidden_size=256,
    output_size=10,
    num_steps=100
)