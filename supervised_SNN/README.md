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

Neuron Models Folder
├── Python Files
│   ├── earlystopping.py
│   ├── models.py
│   └── main.py
└── Subfolders
    ├── alpha
    │   ├── BP_results
    │   ├── SG_results
    │   ├── earlystopping.py
    │   ├── BP_main.py
    │   └── main.py
    ├── leaky
    │   ├── BP_results
    │   ├── SG_results
    │   ├── earlystopping.py
    │   ├── BP_main.py
    │   └── main.py
    ├── synaptic
    │   ├── BP_results
    │   ├── SG_results
    │   ├── earlystopping.py
    │   ├── BP_main.py
    │   └── main.py
    └── lapicque
        ├── BP_results
        ├── SG_results
        ├── earlystopping.py
        ├── BP_main.py
        └── main.py
      

