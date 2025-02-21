# Spiking Neural Network Implementations

This repository contains various implementations of Spiking Neural Networks (SNNs) for different learning paradigms.

## Project Structure

- **NeuroSpike-MNIST**: Implementation of SNN for MNIST dataset classification
- **RSNN**: Deep Reinforcement Learning with SNNs
  - Implementation of CartPole environment using SNNs
  - Includes tutorial notebook and production code
  - Uses policy gradient method with leaky integrate-and-fire neurons
- **SSNN**: Supervised Learning implementations using SNNs

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- SNNTorch
- OpenAI Gym
- NumPy
- Matplotlib

### Installation
```bash
pip install torch snntorch gym numpy matplotlib
```

## Usage

Each implementation has its own directory with specific instructions.

## Development Setup

### Pre-commit Hooks
This project uses pre-commit hooks to maintain code quality. To set up:

- Install pre-commit:
```bash
conda activate rsnn
pre-commit install
```

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]
