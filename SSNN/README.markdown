# SSNN: Supervised Spiking Neural Networks

![SSNN Banner](https://source.unsplash.com/800x200/?neural-network,brain,supervised-learning)  
*Modular framework for supervised learning with Spiking Neural Networks, featuring multiple neuron models and backpropagation methods.*

The **SSNN** project implements a flexible framework for supervised learning using Spiking Neural Networks (SNNs), supporting various neuron models (Leaky Integrate-and-Fire, Alpha, Synaptic, Lapicque) and backpropagation methods (BPTT, RTRL, TBPTT). It includes experiment tracking with Weights & Biases, comprehensive visualization tools, and a robust testing suite. As a Senior AI & Machine Learning Engineer, this project showcases my expertise in neuromorphic computing, supervised learning, and modular software design, complementing my work in reinforcement learning and anomaly detection.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![SNNTorch](https://img.shields.io/badge/SNNTorch-0.7+-FF6F61)](https://snntorch.readthedocs.io/) 
[![Weights & Biases](https://img.shields.io/badge/W&B-2023+-FFBE00?logo=wandb)](https://wandb.ai/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

The **SSNN** framework, part of the `Medium/snn_implementations` repository, provides a modular platform for supervised learning with Spiking Neural Networks. It supports multiple neuron models and backpropagation methods, making it suitable for tasks like MNIST digit classification. The framework integrates with Weights & Biases for experiment tracking and offers advanced visualization tools for spike trains and membrane potentials. This project demonstrates the application of neuromorphic computing to supervised learning, offering a biologically plausible alternative to traditional neural networks.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Multiple Neuron Models** | Supports LIF, Alpha, Synaptic, and Lapicque neurons. | Flexible neuromorphic computing |
| **Backpropagation Methods** | Implements BPTT, RTRL, and TBPTT for training. | Efficient learning strategies |
| **Experiment Tracking** | Integrates with Weights & Biases for metrics logging. | Experiment management |
| **Visualization Tools** | Plots spike trains, membrane potentials, and training metrics. | Debugging and analysis |
| **Unit Testing** | Comprehensive test suite for reliability. | Code quality assurance |
| **Modular Design** | Uses Factory, Strategy, and Template patterns for extensibility. | Custom model development |

---

## 📂 Project Structure

```
Medium/snn_implementations/ssnn/
├── config/                       # Configuration files
│   ├── environment.yml          # Conda environment setup
│   ├── config.yaml             # Experiment parameters
│   └── README.md               # Config documentation
├── models/                       # SNN model implementations
│   ├── base.py                 # Base SNN class
│   ├── neurons.py              # Neuron model definitions
│   └── README.md               # Model documentation
├── training/                     # Training utilities
│   ├── trainer.py              # Training pipeline
│   ├── metrics.py              # Performance metrics
│   └── README.md               # Training documentation
├── utils/                        # Helper functions
│   ├── visualizer.py           # Visualization tools
│   ├── logger.py               # Logging with W&B
│   └── README.md               # Utils documentation
├── tests/                        # Unit tests
│   ├── test_models.py          # Model tests
│   ├── test_training.py        # Training tests
│   └── README.md               # Test documentation
├── results/                      # Experiment outputs
│   ├── plots/                  # Visualization outputs
│   ├── metrics/                # CSV and W&B logs
│   └── README.md               # Results documentation
├── logs/                         # Training logs
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- SNNTorch 0.7+
- Weights & Biases (wandb)
- NumPy, Matplotlib
- Conda (recommended for environment management)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/snn_implementations/ssnn
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda env create -f config/environment.yml
   conda activate ssnn
   ```
   If `environment.yml` is missing, install manually:
   ```bash
   conda create -n ssnn python=3.10
   conda activate ssnn
   conda install pytorch torchvision torchaudio -c pytorch
   pip install -r requirements.txt
   ```

3. **Install Dependencies**:
   Create or update `requirements.txt`:
   ```text
   torch>=2.1.0
   snntorch>=0.7.0
   wandb>=0.15.0
   numpy>=1.23.0
   matplotlib>=3.5.0
   pytest>=7.0.0
   pre-commit>=2.20.0
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Pre-commit Hooks** (optional, for development):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Configure Weights & Biases** (optional):
   ```bash
   wandb login
   ```
   Follow prompts to set up your W&B API key.

**Troubleshooting**:
- **Module Not Found**: Ensure dependencies are installed in the active environment.
- **W&B Errors**: Verify your API key and internet connection.
- **GPU Issues**: Check PyTorch CUDA support (`torch.cuda.is_available()`).

---

## 🖥️ Usage

### Running Experiments
Run the main script to train an SNN model:
```bash
python main.py
```
This trains the model on a default dataset (e.g., MNIST) using the configuration in `config.yaml`.

### Configuring Experiments
Edit `config/config.yaml` to customize the experiment:
```yaml
dataset:
  name: "MNIST"
  path: "data/"
model:
  neuron_type: "LIF"
  input_size: 784
  hidden_size: 256
  output_size: 10
  num_steps: 100
  beta: 0.95
training:
  backprop: "BPTT"
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
visualization:
  enable: true
  log_dir: "results/plots"
wandb:
  project: "ssnn-experiments"
  entity: "your-username"
```

### Visualizing Results
View results in:
- **Local Plots**: Check `results/plots/` for spike trains and metrics.
- **Weights & Biases**: Monitor experiments on the W&B dashboard.
- **CSV Files**: Analyze metrics in `results/metrics/`.
Example visualization code:
```python
from utils.visualizer import Visualizer
visualizer = Visualizer()
visualizer.plot_spike_train(spikes)
visualizer.plot_membrane_potential(membrane_potentials)
```

### Example Usage
1. Train on MNIST with LIF neurons and BPTT:
   ```bash
   python main.py --config config/config.yaml
   ```
2. Test a custom configuration:
   ```bash
   python main.py --config config/custom_config.yaml
   ```

---

## 💻 Development

### Running Tests
Run the test suite:
```bash
pytest tests/
```

### Code Style
Follow PEP 8 guidelines. Run linter:
```bash
flake8 .
```

### Adding Features
To extend the project:
1. Add new neuron models in `models/neurons.py`.
2. Implement new backpropagation methods in `training/trainer.py`.
3. Enhance visualizations in `utils/visualizer.py`.
4. Update tests in `tests/`.

---

## 🤝 Contributing

Contributions to enhance neuron models, backpropagation methods, or visualizations are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/snn_implementations/ssnn
   git checkout -b feature/add-new-neuron
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add Izhikevich neuron model"
   ```
4. **Run Tests and Linting**:
   ```bash
   pytest tests/
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
- Update `README.md`, `ARCHITECTURE.md`, and `requirements.txt` for new features.

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
- [Weights & Biases](https://wandb.ai/) for experiment tracking.
- The neuromorphic computing and supervised learning research community.