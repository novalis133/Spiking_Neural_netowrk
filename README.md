# Spiking Neural Network Implementations

 
*Implementations of Spiking Neural Networks (SNNs) for various learning paradigms, including supervised and reinforcement learning.*

This repository provides a collection of Spiking Neural Network (SNN) implementations designed for researchers, students, and developers interested in neuromorphic computing. It includes projects for MNIST classification, reinforcement learning with CartPole, and supervised learning, leveraging tools like PyTorch and SNNTorch.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![SNNTorch](https://img.shields.io/badge/SNNTorch-0.6+-blue)](https://snntorch.readthedocs.io/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Development Setup](#development-setup)
- [Contributing](#contributing)
- [Tutorials](#tutorials)
- [Status](#status)
- [Contact](#contact)
- [License](#license)

---

## 📂 Project Structure

The repository is organized into distinct directories, each focusing on a specific SNN implementation:

| Directory | Description | Tech Stack |
|-----------|-------------|------------|
| [**NeuroSpike-MNIST**](./NeuroSpike-MNIST) | SNN for classifying the MNIST dataset using leaky integrate-and-fire neurons. Includes a training script and evaluation metrics. | PyTorch, SNNTorch, NumPy, Matplotlib |
| [**RSNN**](./RSNN) | Reinforcement learning with SNNs for the CartPole environment. Features a policy gradient method and a tutorial notebook. | PyTorch, SNNTorch, OpenAI Gym |
| [**SSNN**](./SSNN) | Supervised learning implementations with SNNs, including various neuron models and training strategies. | PyTorch, SNNTorch, NumPy |

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
- **Python**: Version 3.8 or higher
- **PyTorch**: Version 1.9 or higher
- **SNNTorch**: Version 0.6 or higher
- **OpenAI Gym**: For reinforcement learning (RSNN)
- **NumPy**: For numerical computations
- **Matplotlib**: For visualizations

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/spiking-neural-networks.git
   cd spiking-neural-networks
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv snn_env
   source snn_env/bin/activate  # Linux/Mac
   snn_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch>=1.9.0 snntorch>=0.6.0 gym numpy matplotlib
   ```

4. **Verify Installation**:
   Run the following to check if dependencies are installed correctly:
   ```bash
   python -c "import torch, snntorch, gym, numpy, matplotlib; print('All dependencies installed!')"
   ```

**Troubleshooting**:
- Ensure PyTorch is compatible with your CUDA version (if using GPU). Check [PyTorch's official site](https://pytorch.org/get-started/locally/) for installation commands.
- If `gym` installation fails, try `pip install gym==0.21.0` for compatibility.

---

## 🛠️ Usage

Each project directory contains specific instructions in its own `README.md`. Below are quickstart commands:

### NeuroSpike-MNIST
Train and evaluate the SNN on the MNIST dataset:
```bash
cd NeuroSpike-MNIST
python train.py --epochs 10 --batch-size 128
python evaluate.py --model-path models/mnist_snn.pth
```

### RSNN
Run the CartPole reinforcement learning environment:
```bash
cd RSNN
python cartpole_snn.py --episodes 1000
jupyter notebook tutorial.ipynb  # For interactive tutorial
```

### SSNN
Train a supervised SNN model:
```bash
cd SSNN
python supervised_snn.py --dataset custom_dataset --epochs 20
```

**Note**: Check each directory’s `README.md` for detailed configurations and dataset setup.

---

## 🧑‍💻 Development Setup

### Pre-commit Hooks
To maintain code quality, this project uses pre-commit hooks for linting and formatting.

1. **Install pre-commit**:
   ```bash
   pip install pre-commit
   ```

2. **Set Up Hooks**:
   ```bash
   conda activate snn_env  # Or use your virtual environment
   pre-commit install
   ```

3. **Run Hooks**:
   Hooks will automatically run on `git commit` to check code style (e.g., Black, Flake8). To run manually:
   ```bash
   pre-commit run --all-files
   ```

### Code Quality Tools
- **Black**: For code formatting
- **Flake8**: For linting
- **isort**: For import sorting

Install these tools:
```bash
pip install black flake8 isort
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/spiking-neural-networks.git
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Changes**:
   Follow the [Conventional Commits](https://www.conventionalcommits.org/) format, e.g.:
   ```bash
   git commit -m "feat: add new SNN model for time-series data"
   ```

4. **Push and Create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Open a PR on GitHub with a clear description of your changes.

**Guidelines**:
- Ensure code passes pre-commit hooks.
- Include tests for new features (place in `tests/` directory).
- Update documentation in the relevant project directory.

---

## 📖 Tutorials

Explore interactive tutorials in the following directories:
- [RSNN Tutorial Notebook](./RSNN/tutorial.ipynb): Step-by-step guide to implementing SNNs for reinforcement learning.
- [NeuroSpike-MNIST Guide](./NeuroSpike-MNIST/README.md): Detailed instructions for training and evaluating the MNIST model.

---

## 📈 Status

- **NeuroSpike-MNIST**: Stable, actively maintained
- **RSNN**: In development, seeking contributions for new environments
- **SSNN**: Experimental, open to new model architectures

[![Build Status](https://img.shields.io/badge/Build-Passing-green)](https://github.com/Novalis133/spiking-neural-networks/actions)

---

## 📫 Contact

For questions or collaboration:
- **Email**: osama1339669@gmail.com
- **LinkedIn**: [Osama](https://www.linkedin.com/in/osamat339669/)
- **GitHub Issues**: Open an issue on this repository

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
