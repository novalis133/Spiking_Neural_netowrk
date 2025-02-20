
```markdown:%2FUsers%2Fosamaabdelaal%2Fgithub%2Fspinncloud%2Fsrc%2Futils%2FREADME.md
# Utils Module

This module provides utility functions and helper tools for the SNN project.

## Overview

The utils module contains:
- Configuration management
- Visualization tools
- Logging utilities
- Helper functions

## Components

### 1. Config (`config.py`)
Configuration management:
- Model parameters
- Training settings
- Data preprocessing configs
- Experiment tracking

### 2. Visualization (`visualization.py`)
Visualization tools for:
- Spike train visualization
- Network activity plots
- Training progress monitoring
- Performance metrics plotting

## Usage Example

```python
from utils.config import Config
from utils.visualization import plot_spikes

# Load configuration
config = Config('config.yaml')

# Visualize spike trains
plot_spikes(spike_data, title="Neuron Activity")