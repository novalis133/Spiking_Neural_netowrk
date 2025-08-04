# RSNN: Reinforcement Spiking Neural Network

![RSNN Banner](https://source.unsplash.com/800x200/?neural-network,reinforcement-learning,robotics)  
*Deep reinforcement learning framework using Spiking Neural Networks for continuous control tasks.*

The **RSNN** project implements a Spiking Neural Network (SNN) framework for reinforcement learning (RL), designed for continuous control tasks like those in OpenAI Gym environments (e.g., CartPole, Pendulum). It combines biologically plausible SNNs with policy gradient algorithms (REINFORCE, A2C) and advanced visualization tools. As a Senior AI & Machine Learning Engineer, this project showcases my expertise in neuromorphic computing, deep reinforcement learning, and modular software design, complementing my work in MNIST classification and anomaly detection.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![SNNTorch](https://img.shields.io/badge/SNNTorch-0.7+-FF6F61)](https://snntorch.readthedocs.io/) 
[![Gym](https://img.shields.io/badge/Gym-0.26+-199EF6?logo=python)](https://gym.openai.com/) 
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

The **RSNN** framework integrates Spiking Neural Networks (SNNs) with reinforcement learning (RL) to tackle continuous control tasks. It supports multiple neuron models (Leaky Integrate-and-Fire, Alpha, Synaptic) and RL algorithms (REINFORCE, A2C), with a focus on modularity and observability. The project includes environment wrappers for OpenAI Gym, real-time visualization of neural activity, and a robust training pipeline. This work is part of the `Medium/snn_implementations` repository, alongside other SNN projects like NeuroSpike-MNIST and supervised SNN implementations.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Multiple Neuron Models** | Supports LIF, Alpha, and Synaptic neurons. | Flexible neuromorphic computing |
| **RL Algorithms** | Implements REINFORCE and A2C with entropy regularization. | Policy optimization |
| **Environment Support** | Wraps OpenAI Gym environments (e.g., CartPole, Pendulum). | RL benchmarking |
| **3D Visualization** | Real-time 3D plots of neural activity and spike trains. | Debugging and analysis |
| **Modular Design** | Uses Factory, Strategy, and Observer patterns for extensibility. | Custom algorithm development |
| **Real-time Monitoring** | Tracks training metrics and neural states. | Performance evaluation |

---

## 📂 Project Structure

```
Medium/snn_implementations/rsnn/
├── agents/                       # RL agent implementations
│   ├── snn_agent.py             # SNN-based RL agent logic
│   └── README.md                # Agent documentation
├── configs/                      # Configuration files
│   ├── environment.yml          # Conda environment setup
│   ├── config.yaml              # Experiment parameters
│   └── README.md                # Config documentation
├── environments/                 # Environment wrappers
│   ├── gym_wrapper.py           # OpenAI Gym integration
│   └── README.md                # Environment documentation
├── models/                       # SNN model implementations
│   ├── policy_snn.py            # SNN policy definitions
│   └── README.md                # Model documentation
├── training/                     # Training utilities
│   ├── trainer.py               # Training pipeline
│   └── README.md                # Training documentation
├── utils/                        # Helper functions
│   ├── logger.py                # Logging utilities
│   ├── visualizer.py            # Visualization tools
│   └── README.md                # Utils documentation
├── results/                      # Experiment outputs
│   ├── metrics/                 # Training metrics
│   ├── visualizations/          # Neural activity plots
│   └── README.md                # Results documentation
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- SNNTorch 0.7+
- OpenAI Gym 0.26+
- NumPy, Matplotlib
- Conda (recommended for environment management)
- Docker (optional, for containerized deployment)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/snn_implementations/rsnn
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda env create -f configs/environment.yml
   conda activate rsnn
   ```
   If `environment.yml` is missing, install manually:
   ```bash
   conda create -n rsnn python=3.10
   conda activate rsnn
   conda install pytorch torchvision torchaudio -c pytorch
   pip install -r requirements.txt
   ```

3. **Install Dependencies**:
   Create or update `requirements.txt`:
   ```text
   torch>=2.1.0
   snntorch>=0.7.0
   gym==0.26.2
   numpy>=1.23.0
   matplotlib>=3.5.0
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

**Troubleshooting**:
- **Module Not Found**: Ensure dependencies are installed in the active environment.
- **Gym Errors**: Verify `gym==0.26.2` for compatibility.
- **GPU Issues**: Check PyTorch CUDA support (`torch.cuda.is_available()`).

---

## 🖥️ Usage

### Running Experiments
Run the main script to train an SNN agent:
```bash
python main.py
```
This executes the training loop for the default environment (e.g., CartPole) using the REINFORCE algorithm.

### Configuring Experiments
Edit `configs/config.yaml` to customize the experiment:
```yaml
environment:
  name: "CartPole-v1"
  max_steps: 500
model:
  neuron_type: "LIF"
  hidden_size: 128
  num_steps: 100
  beta: 0.95
training:
  algorithm: "REINFORCE"
  learning_rate: 0.001
  episodes: 1000
  batch_size: 32
visualization:
  enable_3d: true
  log_dir: "results/visualizations"
```

### Visualizing Results
Use the visualization tools to monitor training:
```python
from utils.visualizer import Visualizer
visualizer = Visualizer()
visualizer.plot_3d_spike_activity(spikes)
visualizer.plot_training_metrics(metrics)
```

### Example Usage
1. Train on CartPole:
   ```bash
   python main.py --config configs/config.yaml
   ```
2. Train on Pendulum:
   ```bash
   python main.py --config configs/pendulum_config.yaml
   ```

---

## 🐳 Docker Support

1. **Build the Container**:
   ```bash
   docker build -t rsnn .
   ```

2. **Run the Container**:
   ```bash
   docker run -it --gpus all rsnn
   ```

*Note*: Ensure Docker is configured with NVIDIA GPU support.

---

## 💻 Development

### Running Tests
If tests are implemented, run:
```bash
python -m pytest tests/
```
*Note*: Create a `tests/` directory with unit tests for `agents/`, `models/`, and `training/`.

### Code Style
Follow PEP 8 guidelines. Run linter:
```bash
flake8 .
```

### Adding Features
To extend the project:
1. Add new environments in `environments/`.
2. Implement new SNN models in `models/`.
3. Update RL algorithms in `agents/`.
4. Enhance visualizations in `utils/visualizer.py`.

---

## 🤝 Contributing

Contributions to enhance RL algorithms, add environments, or improve visualizations are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/snn_implementations/rsnn
   git checkout -b feature/add-new-algorithm
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add A3C algorithm support"
   ```
4. **Run Tests and Linting**:
   ```bash
   python -m pytest tests/
   flake8 .
   ```
5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-algorithm
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Ensure compatibility with SNNTorch, PyTorch, and Gym.
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
- [OpenAI Gym](https://gym.openai.com/) for RL environments.
- The reinforcement learning and neuromorphic computing research community.