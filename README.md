Spiking Neural Network Implementations
Implementations of Spiking Neural Networks (SNNs) for various learning paradigms, including supervised and reinforcement learning.
This repository provides a collection of Spiking Neural Network (SNN) implementations designed for researchers, students, and developers interested in neuromorphic computing. It includes projects for MNIST classification, reinforcement learning with CartPole, and supervised learning, leveraging tools like PyTorch and SNNTorch.


📚 Table of Contents

Project Structure
Getting Started
Usage
Development Setup
Contributing
Tutorials
Status
Contact
License


📂 Project Structure
The repository is organized into distinct directories, each focusing on a specific SNN implementation:



Directory
Description
Tech Stack



NeuroSpike-MNIST
SNN for classifying the MNIST dataset using leaky integrate-and-fire neurons. Includes a training script and evaluation metrics.
PyTorch, SNNTorch, NumPy, Matplotlib


RSNN
Reinforcement learning with SNNs for the CartPole environment. Features a policy gradient method and a tutorial notebook.
PyTorch, SNNTorch, OpenAI Gym


SSNN
Supervised learning implementations with SNNs, including various neuron models and training strategies.
PyTorch, SNNTorch, NumPy



🚀 Getting Started
Prerequisites
Ensure you have the following installed:

Python: Version 3.8 or higher
PyTorch: Version 1.9 or higher
SNNTorch: Version 0.6 or higher
OpenAI Gym: For reinforcement learning (RSNN)
NumPy: For numerical computations
Matplotlib: For visualizations

Installation

Clone the Repository:
git clone https://github.com/Novalis133/spiking-neural-networks.git
cd spiking-neural-networks


Set Up a Virtual Environment (recommended):
python -m venv snn_env
source snn_env/bin/activate  # Linux/Mac
snn_env\Scripts\activate     # Windows


Install Dependencies:
pip install torch>=1.9.0 snntorch>=0.6.0 gym numpy matplotlib


Verify Installation:Run the following to check if dependencies are installed correctly:
python -c "import torch, snntorch, gym, numpy, matplotlib; print('All dependencies installed!')"



Troubleshooting:

Ensure PyTorch is compatible with your CUDA version (if using GPU). Check PyTorch's official site for installation commands.
If gym installation fails, try pip install gym==0.21.0 for compatibility.


🛠️ Usage
Each project directory contains specific instructions in its own README.md. Below are quickstart commands:
NeuroSpike-MNIST
Train and evaluate the SNN on the MNIST dataset:
cd NeuroSpike-MNIST
python train.py --epochs 10 --batch-size 128
python evaluate.py --model-path models/mnist_snn.pth

RSNN
Run the CartPole reinforcement learning environment:
cd RSNN
python cartpole_snn.py --episodes 1000
jupyter notebook tutorial.ipynb  # For interactive tutorial

SSNN
Train a supervised SNN model:
cd SSNN
python supervised_snn.py --dataset custom_dataset --epochs 20

Note: Check each directory’s README.md for detailed configurations and dataset setup.

🧑‍💻 Development Setup
Pre-commit Hooks
To maintain code quality, this project uses pre-commit hooks for linting and formatting.

Install pre-commit:
pip install pre-commit


Set Up Hooks:
conda activate snn_env  # Or use your virtual environment
pre-commit install


Run Hooks:Hooks will automatically run on git commit to check code style (e.g., Black, Flake8). To run manually:
pre-commit run --all-files



Code Quality Tools

Black: For code formatting
Flake8: For linting
isort: For import sorting

Install these tools:
pip install black flake8 isort


🤝 Contributing
Contributions are welcome! To contribute:

Fork the Repository:
git fork https://github.com/Novalis133/spiking-neural-networks.git


Create a Feature Branch:
git checkout -b feature/your-feature-name


Commit Changes:Follow the Conventional Commits format, e.g.:
git commit -m "feat: add new SNN model for time-series data"


Push and Create a Pull Request:
git push origin feature/your-feature-name

Open a PR on GitHub with a clear description of your changes.


Guidelines:

Ensure code passes pre-commit hooks.
Include tests for new features (place in tests/ directory).
Update documentation in the relevant project directory.


📖 Tutorials
Explore interactive tutorials in the following directories:

RSNN Tutorial Notebook: Step-by-step guide to implementing SNNs for reinforcement learning.
NeuroSpike-MNIST Guide: Detailed instructions for training and evaluating the MNIST model.


📈 Status

NeuroSpike-MNIST: Stable, actively maintained
RSNN: In development, seeking contributions for new environments
SSNN: Experimental, open to new model architectures



📫 Contact
For questions or collaboration:

Email: osama1339669@gmail.com
LinkedIn: Osama
GitHub Issues: Open an issue on this repository


📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
