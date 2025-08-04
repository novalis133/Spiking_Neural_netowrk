# Spiking Neural Network Implementations

*Biologically-inspired Spiking Neural Networks for MNIST classification, reinforcement learning, and supervised learning tasks.*

This repository hosts a collection of Spiking Neural Network (SNN) implementations, showcasing efficient and biologically plausible neural processing for various learning paradigms. The primary focus is **NeuroSpike-MNIST**, a robust SNN for MNIST digit classification, alongside implementations for deep reinforcement learning (DRL) in the CartPole environment and supervised learning tasks. As a Senior AI & Machine Learning Engineer, this project highlights my expertise in neuromorphic computing, deep reinforcement learning, and AI-driven tool development, complementing my work in robotics and anomaly detection.

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
- [Docker Support](#docker-support)
- [Development](#development)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

This repository contains implementations of Spiking Neural Networks (SNNs) using SNNTorch and PyTorch, designed for:
- **NeuroSpike-MNIST**: MNIST digit classification using Leaky Integrate-and-Fire (LIF) neurons with rate and temporal coding.
- **drl_SNN**: Deep reinforcement learning with SNNs for the CartPole environment, using policy gradient methods.
- **supervised_SNN**: Supervised learning implementations with customizable SNN architectures.

The **NeuroSpike-MNIST** project is the flagship implementation, featuring a modular architecture, GPU-accelerated computation, and advanced visualization tools for spike trains and membrane potentials. It demonstrates biologically plausible neural processing and aligns with my expertise in neuromorphic computing and AI-driven solutions.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **LIF Neurons** | Implements Leaky Integrate-and-Fire neurons for biological plausibility. | Neuromorphic computing |
| **Spike Encoding** | Supports rate and temporal coding for flexible data processing. | MNIST classification |
| **GPU Acceleration** | Leverages PyTorch for fast computation on GPUs. | Scalable training |
| **Visualization Tools** | Real-time plots of spike trains and membrane potentials. | Debugging and analysis |
| **Modular Architecture** | Configurable network topology and training pipeline. | Custom model development |
| **DRL Support** | Policy gradient methods for CartPole in `drl_SNN`. | Reinforcement learning |

---

## 📂 Project Structure

```
Medium/snn_implementations/
├── NeuroSpike-MNIST/             # MNIST classification with SNNs
│   ├── src/                     # Source code
│   │   ├── data/                # Data loading and preprocessing
│   │   │   ├── data_loader.py
│   │   │   ├── preprocessor.py
│   │   │   └── README.md
│   │   ├── models/              # SNN model implementations
│   │   │   ├── snn_model.py
│   │   │   ├── layers.py
│   │   │   └── README.md
│   │   ├── training/            # Training and evaluation
│   │   │   ├── trainer.py
│   │   │   ├── metrics.py
│   │   │   └── README.md
│   │   ├── utils/               # Utilities and visualization
│   │   │   ├── config.py
│   │   │   ├── visualization.py
│   │   │   └── README.md
│   │   └── main.py              # Main execution script
│   ├── config.yaml              # Configuration file
│   ├── requirements.txt         # Dependencies
│   ├── Dockerfile               # Container configuration
│   └── README.md                # Subproject documentation
├── drl_SNN/                     # Deep reinforcement learning with SNNs
│   ├── cartpole_snn.py          # CartPole implementation
│   ├── tutorial.ipynb           # Tutorial notebook
│   └── README.md                # Subproject documentation
├── supervised_SNN/              # Supervised learning with SNNs
│   ├── src/                     # Source code (TBD)
│   └── README.md                # Subproject documentation
├── LICENSE                      # MIT License
└── README.md                    # Main project documentation
```

*Note*: Some directories (e.g., `supervised_SNN/src`) are placeholders and will be populated as implementations are completed.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- SNNTorch 0.7+
- OpenAI Gym (for `drl_SNN`)
- NumPy, Matplotlib
- Conda (recommended for environment management)
- Docker (optional, for containerized deployment)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/snn_implementations
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda create -n snn_env python=3.10
   conda activate snn_env
   ```

3. **Install PyTorch**:
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r NeuroSpike-MNIST/requirements.txt
   ```
   *Note*: If `requirements.txt` is missing, install manually:
   ```bash
   pip install snntorch gym numpy matplotlib
   ```

5. **Set Up Pre-commit Hooks** (optional, for development):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

**Troubleshooting**:
- **Module Not Found**: Ensure all dependencies are installed in the active environment.
- **GPU Issues**: Verify PyTorch is installed with CUDA support (`torch.cuda.is_available()`).
- **File Paths**: Run commands from the `snn_implementations` directory.

---

## 🖥️ Usage

### NeuroSpike-MNIST
1. **Configure the Model**:
   Edit `NeuroSpike-MNIST/config.yaml`:
   ```yaml
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

2. **Train a Model**:
   ```bash
   cd NeuroSpike-MNIST
   python src/main.py --mode train
   ```

3. **Run Inference**:
   ```bash
   python src/main.py --mode inference --model_path checkpoints/best_model.pth --image_path examples/digit_5.png
   ```

4. **Visualize Results**:
   Use the visualization tools:
   ```python
   from src.utils.visualization import SpikeVisualizer
   visualizer = SpikeVisualizer()
   visualizer.plot_spike_train(spikes)
   visualizer.plot_membrane_potential(membrane_potentials)
   ```

### drl_SNN
1. **Run CartPole Training**:
   ```bash
   cd drl_SNN
   python cartpole_snn.py
   ```

2. **Explore Tutorial**:
   Open `tutorial.ipynb` in Jupyter Notebook for a guided walkthrough.

### supervised_SNN
*Note*: This module is under development. Check `supervised_SNN/README.md` for updates.

---

## 🐳 Docker Support

1. **Build the Container**:
   ```bash
   cd NeuroSpike-MNIST
   docker build -t snn_mnist .
   ```

2. **Run the Container**:
   ```bash
   docker run -it --gpus all snn_mnist
   ```

*Note*: Ensure Docker is configured with GPU support for NVIDIA GPUs.

---

## 💻 Development

### Running Tests
```bash
cd NeuroSpike-MNIST
python -m pytest tests/
```

### Code Style
Follow PEP 8 guidelines. Run linter:
```bash
flake8 src/
```

### Adding New Implementations
To add a new SNN implementation:
1. Create a new directory (e.g., `new_snn_project/`).
2. Follow the structure of `NeuroSpike-MNIST/src/`.
3. Update `requirements.txt` and `README.md`.

---

## 🤝 Contributing

Contributions to enhance SNN implementations, add new paradigms, or improve visualizations are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/snn_implementations
   git checkout -b feature/add-new-snn
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new SNN implementation"
   ```
4. **Run Tests and Linting**:
   ```bash
   python -m pytest tests/
   flake8 src/
   ```
5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-snn
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
- [OpenAI Gym](https://gym.openai.com/) for reinforcement learning environments.
- The neural coding and SNN research community.
