# SNN Reinforcement Learning for CartPole

*Spiking Neural Network implementation for reinforcement learning in the CartPole environment using policy gradient methods.*

This project implements a Spiking Neural Network (SNN) for reinforcement learning, specifically applied to the CartPole environment from OpenAI Gym. It utilizes Leaky Integrate-and-Fire (LIF) neurons and a policy gradient method to train an agent to balance a pole on a cart. As a Senior AI & Machine Learning Engineer, this project showcases my expertise in neuromorphic computing, deep reinforcement learning, and biologically-inspired AI, complementing my work in MNIST classification and anomaly detection.

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
- [Development](#development)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

This project implements a Spiking Neural Network (SNN) for reinforcement learning in the CartPole environment. The SNN uses Leaky Integrate-and-Fire (LIF) neurons and a policy gradient method to learn an optimal policy for balancing the pole. The implementation includes a modular policy network, a training loop, and a Jupyter notebook with detailed explanations of the SNN backpropagation algorithm. This work demonstrates the application of neuromorphic computing to reinforcement learning, offering a biologically plausible alternative to traditional neural networks.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **LIF Neurons** | Uses Leaky Integrate-and-Fire neurons for spike-based processing. | Neuromorphic computing |
| **Policy Gradient** | Implements REINFORCE algorithm for policy optimization. | Reinforcement learning |
| **CartPole Environment** | Trains an agent to balance a pole in OpenAI Gym’s CartPole. | RL benchmarking |
| **Visualization** | Plots episode rewards and spike activity. | Training analysis |
| **Tutorial Notebook** | Detailed explanation of SNN backpropagation. | Educational resource |
| **Modular Design** | Separates policy network and training logic. | Extensible development |

---

## 📂 Project Structure

```
Medium/snn_implementations/drl_SNN/
├── run.py                        # Main script for training and evaluation
├── s_policy.py                   # Policy network definition and RL logic
├── SNN_BP_Part_3.ipynb           # Tutorial notebook on SNN backpropagation
├── requirements.txt              # Project dependencies
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
- Jupyter Notebook (for `SNN_BP_Part_3.ipynb`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/snn_implementations/drl_SNN
   ```

2. **Create and Activate a Conda Environment**:
   ```bash
   conda create -n drl_snn python=3.10
   conda activate drl_snn
   ```

3. **Install PyTorch**:
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install manually:
   ```bash
   pip install snntorch gym==0.26.2 numpy matplotlib jupyter
   ```

**Troubleshooting**:
- **Module Not Found**: Ensure dependencies are installed in the active environment.
- **Gym Errors**: Verify `gym==0.26.2` for compatibility with CartPole.
- **Jupyter Issues**: Run `jupyter notebook` from the project directory to open `SNN_BP_Part_3.ipynb`.

---

## 🖥️ Usage

### Training the Agent
Run the main script to train the SNN agent:
```bash
python run.py
```
This executes the training loop, using the policy gradient method to optimize the SNN for the CartPole environment. Training progress (e.g., episode rewards) is visualized using Matplotlib.

### Exploring the Tutorial
Open the Jupyter notebook for a detailed explanation of the SNN backpropagation algorithm:
```bash
jupyter notebook SNN_BP_Part_3.ipynb
```
The notebook includes code, visualizations, and explanations of the LIF neuron model and policy gradient training.

### Customizing the Policy
Modify `s_policy.py` to adjust the SNN architecture (e.g., number of neurons, layers) or policy gradient hyperparameters (e.g., learning rate). Example:
```python
from s_policy import SNNPolicy
policy = SNNPolicy(
    input_size=4,  # CartPole state space
    hidden_size=128,
    output_size=2,  # CartPole action space
    num_steps=100,  # Time steps for spiking
    beta=0.95  # LIF decay factor
)
```

### Visualizing Results
Use Matplotlib to plot training metrics (e.g., episode rewards, spike trains):
```python
import matplotlib.pyplot as plt
plt.plot(rewards)  # rewards from training
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
```

---

## 💻 Development

### Running Tests
If tests are implemented, run:
```bash
python -m pytest tests/
```
*Note*: Create a `tests/` directory with unit tests for `run.py` and `s_policy.py` if needed.

### Code Style
Follow PEP 8 guidelines. Run linter:
```bash
pip install flake8
flake8 .
```

### Adding Features
To extend the project (e.g., add new environments or SNN models):
1. Update `s_policy.py` with new policy definitions.
2. Modify `run.py` to support additional environments (e.g., LunarLander).
3. Add visualizations in `SNN_BP_Part_3.ipynb`.

---

## 🤝 Contributing

Contributions to enhance the SNN implementation, add new RL environments, or improve the tutorial are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/snn_implementations/drl_SNN
   git checkout -b feature/add-new-environment
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add LunarLander environment support"
   ```
4. **Run Tests and Linting**:
   ```bash
   python -m pytest tests/
   flake8 .
   ```
5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-environment
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Ensure compatibility with SNNTorch, PyTorch, and Gym.
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
- [OpenAI Gym](https://gym.openai.com/) for the CartPole environment.
- The reinforcement learning and neuromorphic computing research community.
