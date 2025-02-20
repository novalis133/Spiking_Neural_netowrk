# NeuroSpike-MNIST: Biologically-Inspired Digit Recognition

A comprehensive implementation of Spiking Neural Networks using SNNTorch, designed for efficient and flexible neural processing.

## Project Objective

NeuroSpike-MNIST aims to provide a robust implementation of Spiking Neural Networks (SNNs) for MNIST digit classification. The project demonstrates:

🧠 **Biological Inspiration**
- Implementation of Leaky Integrate-and-Fire (LIF) neurons
- Support for rate and temporal coding methods
- Biologically plausible neural processing

🎯 **Core Features**
- MNIST digit classification using spike-based processing
- Real-time visualization of neuron activities
- Configurable network architecture
- Comprehensive performance metrics

🔧 **Technical Implementation**
- Modular and extensible architecture
- GPU-accelerated computation support
- Advanced visualization tools
- Flexible data preprocessing pipeline

## Project Overview

This project implements a Spiking Neural Network (SNN) framework with the following features:
- Flexible network architecture
- Multiple spike encoding methods
- Comprehensive visualization tools
- Configurable training pipeline

## Project Structure

├── src/
│   ├── data/           # Data loading and preprocessing
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── README.md
│   ├── models/         # SNN model implementations
│   │   ├── snn_model.py
│   │   ├── layers.py
│   │   └── README.md
│   ├── training/       # Training and evaluation
│   │   ├── trainer.py
│   │   ├── metrics.py
│   │   └── README.md
│   ├── utils/          # Utilities and visualization
│   │   ├── config.py
│   │   ├── visualization.py
│   │   └── README.md
│   └── main.py         # Main execution script
├── config.yaml         # Configuration file
├── requirements.txt    # Project dependencies
├── Dockerfile         # Container configuration
└── README.md          # Project documentation


## Key Components

### Data Module
- Flexible data loading pipeline
- Multiple spike encoding strategies (Rate and Temporal coding)
- Customizable preprocessing transformations

### Models Module
- Leaky Integrate-and-Fire (LIF) neuron implementation
- Residual connections for deep architectures
- Configurable network topology
- State management for spiking neurons

### Training Module
- Comprehensive training pipeline
- Multiple metrics tracking (Accuracy, Spike Rate)
- Checkpoint management
- Learning rate scheduling

### Utils Module
- YAML-based configuration management
- Visualization tools for spike trains
- Membrane potential monitoring
- Training progress visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spinncloud.git
cd spinncloud
```
2. Create and activate conda environment (recommended):
```bash
conda create -n spinncloud python=3.10
conda activate spinncloud
```
3. Install PyTorch and related packages:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
## Configuration
```yaml
Create or modify config.yaml :
model:
  input_size: 784
  hidden_size: 256
  output_size: 10
  num_steps: 100
  beta: 0.95

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999

data_path: "data/"
```
## Usage

### Training Mode
To train a new model:
```bash
python src/main.py --mode train
```
### Inference Mode
To use a pretrained model for prediction:
```bash
python src/main.py --mode inference --model_path checkpoints/best_model.pth --image_path path/to/digit.png
```

### Command Line Arguments
- --mode : Choose between 'train' or 'inference' mode (default: 'train')
- --model_path : Path to pretrained model for inference (default: 'checkpoints/best_model.pth')
- --image_path : Path to input image for inference (required for inference mode)

### Example Usage
1. Train a new model:
```bash
python src/main.py --mode train
```
2. Predict using pretrained model:
```bash
python src/main.py --mode inference \
                   --model_path checkpoints/best_model.pth \
                   --image_path examples/digit_5.png
```

## Custom Models Creation
```python
from src.models.snn_model import SNNModel
from src.training.trainer import SNNTrainer
```
# Initialize model
```python
model = SNNModel(
    input_size=784,
    hidden_size=256,
    output_size=10
)
```

# Create trainer
```python
trainer = SNNTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion
)
```

## Visualization
```python
from src.utils.visualization import SpikeVisualizer

visualizer = SpikeVisualizer()
visualizer.plot_spike_train(spikes)
visualizer.plot_membrane_potential(membrane_potentials)
```

## Docker Support
1. Build the container:
```bash
docker build -t spinncloud .
```

2. Run the container:
```bash
docker run -it --gpus all spinncloud
```

## Development
### Running Tests
```bash
python -m pytest tests/
```

### Code Style
We follow PEP 8 guidelines. Run linter:
```bash
flake8 src/
```

## Contributing
1. Fork the repository
2. Create your feature branch ( git checkout -b feature/amazing-feature )
3. Commit your changes ( git commit -m 'Add amazing feature' )
4. Push to the branch ( git push origin feature/amazing-feature )
5. Open a Pull Request
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- SNNTorch library and its contributors
- PyTorch team
- Neural coding and SNN research community

## Contact
