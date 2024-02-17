
# Project Title

## Overview
This project implements a Spiking Neural Network (SNN) for reinforcement learning, particularly applied to the CartPole environment from OpenAI's gym. It utilizes leaky integrate-and-fire neurons and a policy gradient method for training.

## Files
- `run.py`: Main script that sets up the environment, defines the neural network model, and executes the training loop.
- `s_policy.py`: Contains the definition of the policy network used by the SNN agent, including action selection and reinforcement learning dataset preparation.

## Requirements
- Python 3.x
- PyTorch
- gym
- snntorch

## Installation
To set up the environment, run:
\```bash
pip install torch gym snntorch
\```

## Usage
Run the main script to start training:
\```bash
python run.py
\```

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
