# Neuron Models Folder

This repo contains code for testing neuron models using different backpropagation algorithms and surrogate functions.

## Python Files

- `earlystopping.py`: contains code for early stopping of the model training
- `models.py`: contains code for the main helper functions used for all training files.
- `main.py`: contains code for testing the neuron models using different surrogate functions and the Backpropagation algorithm used, not a run file.

## Subfolders

- `alpha`: contains code for testing the alpha neuron model.
  - `BP_results`: contains results of tests using backpropagation algorithms.
  - `SG_results`: contains results of tests using Surrogate functions.
  - `earlystopping.py`: contains code for early stopping of the model training
  - `BP_main.py`: contains code to run the experiments to evaluate the SNN for backpropagation algorithms.
  - `main.py`: contains code for testing the neuron models using different surrogate functions with BPTT.
- `leaky`: contains code for testing the neuron model with backpropagation algorithm 2
  - `BP_results`: contains results of tests using backpropagation algorithms.
  - `SG_results`: contains results of tests using Surrogate functions.
  - `earlystopping.py`: contains code for early stopping of the model training
  - `BP_main.py`: contains code to run the experiments to evaluate the SNN for backpropagation algorithms.
  - `main.py`: contains code for testing the neuron models using different surrogate functions with BPTT.
- `synaptic`: contains code for testing the neuron model with backpropagation algorithm 3
  - `BP_results`: contains results of tests using backpropagation algorithms.
  - `SG_results`: contains results of tests using Surrogate functions.
  - `earlystopping.py`: contains code for early stopping of the model training
  - `BP_main.py`: contains code to run the experiments to evaluate the SNN for backpropagation algorithms.
  - `main.py`: contains code for testing the neuron models using different surrogate functions with BPTT.
- `lapicque`: contains code for testing the neuron model with backpropagation algorithm 4
  - `BP_results`: contains results of tests using backpropagation algorithms.
  - `SG_results`: contains results of tests using Surrogate functions.
  - `earlystopping.py`: contains code for early stopping of the model training
  - `BP_main.py`: contains code to run the experiments to evaluate the SNN for backpropagation algorithms.
  - `main.py`: contains code for testing the neuron models using different surrogate functions with BPTT.

## Plotting

- `plotting.py`: contains code for aggregating the produced .csv files and creating plots

## Requirements

- Python 3.8 or higher
- PyTorch 1.11 cuda XX or higher
- snntorch 'pip install snntorch'

## Neuron Models Folder Structure

- **Python Files**
  - `earlystopping.py`
  - `models.py`
  - `main.py`
- **Subfolders**
  - **alpha**
    - `BP_results`, `SG_results`
    - `earlystopping.py`, `BP_main.py`, `main.py`
  - **leaky**
    - `BP_results`, `SG_results`
    - `earlystopping.py`, `BP_main.py`, `main.py`
  - **synaptic**
    - `BP_results`, `SG_results`
    - `earlystopping.py`, `BP_main.py`, `main.py`
  - **lapicque**
    - `BP_results`, `SG_results`
    - `earlystopping.py`, `BP_main.py`, `main.py`

  # SNN Models Documentation

## Network Architecture
- Input Layer: 784 neurons (28x28 MNIST images)
- First Conv Layer: 12 channels, 5x5 kernel
- Second Conv Layer: 64 channels, 5x5 kernel
- Output Layer: 10 neurons (digit classes)

## Neuron Parameters
- Beta (decay factor): 0.5
- Alpha (synaptic decay): 0.9
- Slope: 50
- Time steps: 25

## Surrogate Gradients
- FastSigmoid
- Triangular
- Sigmoid
- SparseFastSigmoid
- SpikeRateEscape
- StochasticSpikeOperator
- StraightThroughEstimator
- ATan

# Training Documentation

## Configuration
- Batch Size: 128
- Learning Rate: 1e-2
- Optimizer: Adamax
- Loss Function: Cross-Entropy Rate Loss
- Early Stopping: 10 epochs patience

## Training Process
1. Forward Pass
   - Processes input through convolutional layers
   - Applies neuron dynamics
   - Records spike and membrane potentials

2. Backward Pass
   - Uses specified surrogate gradient
   - Implements chosen backpropagation method
   - Updates weights using Adamax optimizer

3. Evaluation
   - Tracks training and test accuracy
   - Records results in CSV files
   - Implements early stopping

