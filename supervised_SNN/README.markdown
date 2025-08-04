# Neuron Models: Spiking Neural Network Evaluation

*Framework for evaluating Spiking Neural Network neuron models with various backpropagation algorithms and surrogate gradients.*

The **Neuron Models** project provides a modular framework for testing Spiking Neural Network (SNN) neuron models (Alpha, Leaky, Synaptic, Lapicque) using different backpropagation algorithms (e.g., BPTT) and surrogate gradient functions (e.g., FastSigmoid, ATan). Designed for supervised learning tasks like MNIST digit classification, it includes tools for experiment tracking, visualization, and result aggregation. As a Senior AI & Machine Learning Engineer, this project showcases my expertise in neuromorphic computing, supervised learning, and rigorous model evaluation, complementing my work in reinforcement learning and anomaly detection.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![SNNTorch](https://img.shields.io/badge/SNNTorch-0.7+-FF6F61)](https://snntorch.readthedocs.io/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Visualizing Results](#visualizing-results)
- [Development](#development)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

The **Neuron Models** project, part of the `Spiking_Neural_network/supervised_SNN` repository, is designed to evaluate the performance of various SNN neuron models on supervised learning tasks, primarily MNIST digit classification. It supports multiple neuron models (Alpha, Leaky, Synaptic, Lapicque), backpropagation methods (e.g., BPTT), and surrogate gradient functions (e.g., FastSigmoid, ATan). The framework includes early stopping, comprehensive visualization tools, and CSV-based result logging, enabling systematic comparison of neuron performance. This project highlights my expertise in neuromorphic computing and rigorous model evaluation.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Multiple Neuron Models** | Supports Alpha, Leaky, Synaptic, and Lapicque neurons. | Neuromorphic model evaluation |
| **Backpropagation Methods** | Implements BPTT with customizable surrogate gradients. | Flexible training strategies |
| **Surrogate Gradients** | Supports FastSigmoid, Triangular, ATan, and more. | Gradient approximation for spikes |
| **Early Stopping** | Stops training after 10 epochs of no improvement. | Efficient training |
| **Visualization Tools** | Aggregates results and plots metrics (accuracy, spikes). | Performance analysis |
| **Experiment Tracking** | Logs results to CSV for comparison across models. | Systematic evaluation |

---

## 📂 Project Structure

```
Medium/snn_implementations/neuron_models/
├── alpha/                        # Alpha neuron experiments
│   ├── BP_results/              # Backpropagation results (CSV)
│   ├── SG_results/              # Surrogate gradient results (CSV)
│   ├── earlystopping.py         # Early stopping logic
│   ├── BP_main.py               # Run backpropagation experiments
│   ├── main.py                  # Test with surrogate gradients and BPTT
│   └── README.md                # Alpha-specific documentation
├── leaky/                        # Leaky neuron experiments
│   ├── BP_results/              # Backpropagation results (CSV)
│   ├── SG_results/              # Surrogate gradient results (CSV)
│   ├── earlystopping.py         # Early stopping logic
│   ├── BP_main.py               # Run backpropagation experiments
│   ├── main.py                  # Test with surrogate gradients and BPTT
│   └── README.md                # Leaky-specific documentation
├── synaptic/                     # Synaptic neuron experiments
│   ├── BP_results/              # Backpropagation results (CSV)
│   ├── SG_results/              # Surrogate gradient results (CSV)
│   ├── earlystopping.py         # Early stopping logic
│   ├── BP_main.py               # Run backpropagation experiments
│   ├── main.py                  # Test with surrogate gradients and BPTT
│   └── README.md                # Synaptic-specific documentation
├── lapicque/                     # Lapicque neuron experiments
│   ├── BP_results/              # Backpropagation results (CSV)
│   ├── SG_results/              # Surrogate gradient results (CSV)
│   ├── earlystopping.py         # Early stopping logic
│   ├── BP_main.py               # Run backpropagation experiments
│   ├── main.py                  # Test with surrogate gradients and BPTT
│   └── README.md                # Lapicque-specific documentation
├── earlystopping.py              # Shared early stopping logic
├── models.py                     # Shared helper functions for SNN models
├── main.py                       # Shared testing code (not for running)
├── plotting.py                   # Result aggregation and plotting
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (3.8+ supported, but 3.10 recommended)
- PyTorch 2.1+ (with CUDA for GPU support)
- SNNTorch 0.7+
- NumPy, Matplotlib
- Conda (recommended for environment management)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/snn_implementations/neuron_models
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda create -n neuron_models python=3.10
   conda activate neuron_models
   ```

3. **Install PyTorch**:
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

4. **Install Dependencies**:
   Create or update `requirements.txt`:
   ```text
   torch>=2.1.0
   snntorch>=0.7.0
   numpy>=1.23.0
   matplotlib>=3.5.0
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

**Troubleshooting**:
- **Module Not Found**: Ensure dependencies are installed in the active environment.
- **GPU Issues**: Verify PyTorch CUDA support (`torch.cuda.is_available()`).
- **File Paths**: Run commands from the `neuron_models` directory.

---

## 🖥️ Usage

### Running Experiments
1. **Test a Neuron Model**:
   Run experiments for a specific neuron model (e.g., Alpha) with backpropagation:
   ```bash
   cd alpha
   python BP_main.py
   ```
   Test with surrogate gradients and BPTT:
   ```bash
   python main.py
   ```

2. **Configure Experiments**:
   Each subfolder (`alpha`, `leaky`, etc.) may include a configuration file (e.g., `config.yaml`). Example:
   ```yaml
   dataset:
     name: "MNIST"
     path: "data/"
   model:
     neuron_type: "Alpha"
     input_size: 784
     hidden_size: 256
     output_size: 10
     num_steps: 25
     beta: 0.5
     alpha: 0.9
     slope: 50
   training:
     backprop: "BPTT"
     batch_size: 128
     learning_rate: 0.01
     optimizer: "Adamax"
     loss: "CrossEntropyRate"
     early_stopping_patience: 10
   ```

3. **Supported Backpropagation Methods**:
   - Backpropagation Through Time (BPTT) with surrogate gradients.
   - Specific algorithms vary by subfolder (e.g., "algorithm 2" for Leaky).

4. **Supported Surrogate Gradients**:
   - FastSigmoid, Triangular, Sigmoid, SparseFastSigmoid, SpikeRateEscape, StochasticSpikeOperator, StraightThroughEstimator, ATan.

### Visualizing Results
Aggregate and plot results using:
```bash
python plotting.py
```
This generates plots from CSV files in `BP_results/` and `SG_results/` for each neuron model, saved in the respective subfolder.

Example plotting code:
```python
from plotting import Plotter
plotter = Plotter()
plotter.aggregate_results("alpha/BP_results")
plotter.plot_metrics(output_dir="alpha/plots")
```

---

## 📊 Visualizing Results

- **Local Plots**: Check `alpha/plots/`, `leaky/plots/`, etc., for accuracy, loss, and spike activity plots.
- **CSV Files**: Results are saved in `BP_results/` and `SG_results/` as CSV files for custom analysis.
- **Visualization Types**:
  - Training/test accuracy over epochs.
  - Spike rates and membrane potentials.
  - Comparison of surrogate gradient performance.

---

## 💻 Development

### Network Architecture
- **Input Layer**: 784 neurons (28x28 MNIST images).
- **First Conv Layer**: 12 channels, 5x5 kernel.
- **Second Conv Layer**: 64 channels, 5x5 kernel.
- **Output Layer**: 10 neurons (digit classes).

### Neuron Parameters
- **Beta (decay factor)**: 0.5
- **Alpha (synaptic decay)**: 0.9 (for Alpha neurons)
- **Slope**: 50 (for surrogate gradients)
- **Time Steps**: 25

### Training Process
1. **Forward Pass**:
   - Processes MNIST images through convolutional layers.
   - Applies neuron dynamics (e.g., LIF, Alpha).
   - Records spikes and membrane potentials.
2. **Backward Pass**:
   - Uses specified surrogate gradient (e.g., FastSigmoid).
   - Implements BPTT or other backpropagation methods.
   - Updates weights with Adamax optimizer.
3. **Evaluation**:
   - Tracks accuracy and loss.
   - Saves results to CSV in `BP_results/` or `SG_results/`.
   - Applies early stopping (10 epochs patience).

### Running Tests
No test suite is provided, but you can create one:
```bash
mkdir tests
pytest tests/
```

### Code Style
Follow PEP 8 guidelines. Run linter:
```bash
pip install flake8
flake8 .
```

### Adding Features
To extend the project:
1. Add new neuron models in `models.py` or subfolder-specific files.
2. Implement new surrogate gradients in `main.py` or `BP_main.py`.
3. Enhance plotting in `plotting.py`.

---

## 🤝 Contributing

Contributions to add neuron models, surrogate gradients, or visualization tools are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/snn_implementations/neuron_models
   git checkout -b feature/add-new-neuron
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add Izhikevich neuron model"
   ```
4. **Run Linting**:
   ```bash
   flake8 .
   ```
5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-neuron
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Ensure compatibility with SNNTorch and PyTorch.
- Update `README.md` and `requirements.txt` for new features.

---

## 📫 Contact

- **Email**: osama1339669@gmail.com
- **LinkedIn**: [Osama](https://www.linkedin.com/in/osamat339669/)
- **GitHub Issues**: [Issues Page](https://github.com/Novalis133/Medium/issues)
- **Medium Blog**: [Osama’s Medium](https://medium.com/@osama1339669)

---

## 🙏 Acknowledgments

- [SNNTorch](https://snntorch.readthedocs.io/) for SNN support.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for evaluation.
- The neuromorphic computing and supervised learning research community.
