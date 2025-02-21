# Supervised Spiking Neural Networks (SSNN)

Implementation of various Spiking Neural Network architectures with different neuron models and training methods.

## Features

- Multiple neuron models (Leaky, Alpha, Synaptic, Lapicque)
- Various backpropagation methods (BPTT, RTRL, TBPTT)
- Experiment tracking with Weights & Biases
- Comprehensive visualization tools
- Unit testing suite

## Project Structure
```markdown
SSNN/
├── config/         # Configuration files
├── models/         # Neural network implementations
├── training/       # Training utilities
├── utils/          # Helper functions
├── tests/          # Unit tests
├── results/        # Experiment results
└── logs/          # Training logs
```


## Setup

1. Clone the repository
2. Run setup script:
```bash
chmod +x setup.sh
./setup.sh
```

## Running Experiments
```bash
python main.py
```

## Visualizing Results
Results are automatically saved in the results directory and can be viewed:

- Locally using generated plots
- On Weights & Biases dashboard
- In CSV format for custom analysis

## Testing
Run the test suite:
```bash
pytest tests/
```

## Development Setup

### Pre-commit Hooks
This project uses pre-commit hooks to maintain code quality. To set up:

1. Install pre-commit:
```bash
conda activate rsnn
pre-commit install
```