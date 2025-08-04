# RSNN Architecture Documentation

![RSNN Architecture Banner](https://source.unsplash.com/800x200/?software-architecture,neural-network)  
*Technical documentation for the Reinforcement Spiking Neural Network framework.*

This document details the architecture of the **RSNN** framework, a Spiking Neural Network (SNN) system for reinforcement learning (RL) in continuous control tasks. It outlines design patterns, system components, data flow, and algorithms, emphasizing modularity, extensibility, and observability. This complements the user-facing `README.md` and aligns with my expertise in neuromorphic computing and RL.

---

## 📚 Table of Contents

- [Design Patterns](#design-patterns)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [Algorithms](#algorithms)
- [Design Principles](#design-principles)
- [System Features](#system-features)
- [Detailed Implementations](#detailed-implementations)
- [System Diagrams](#system-diagrams)

---

## 📐 Design Patterns

The RSNN framework employs design patterns to ensure modularity and extensibility:

1. **Factory Pattern** (`models/policy_snn.py`):
   - Creates SNN policies (e.g., LIF, Alpha, Synaptic) based on configuration.
   - Enables dynamic selection of neuron types and architectures.

2. **Strategy Pattern** (`agents/snn_agent.py`):
   - Supports interchangeable components:
     - RL algorithms (REINFORCE, A2C).
     - Action selection strategies (e.g., Epsilon-Greedy, Softmax).
     - Reward processing methods (e.g., discounted returns).

3. **Observer Pattern** (`utils/logger.py`, `utils/visualizer.py`):
   - Monitors real-time metrics:
     - Training progress (rewards, losses).
     - Neural activity (spike trains, membrane potentials).
     - 3D visualizations of network states.

4. **Command Pattern** (`training/trainer.py`):
   - Encapsulates training operations (e.g., policy updates, experience collection).
   - Supports multiple training strategies and experiment workflows.

---

## 🏗️ System Architecture

### Component Structure
```
Medium/snn_implementations/rsnn/
├── agents/                       # RL agent logic
│   └── snn_agent.py             # SNN-based RL agent
├── configs/                      # Configuration management
│   ├── environment.yml          # Conda environment
│   └── config.yaml              # Experiment parameters
├── environments/                 # Environment wrappers
│   └── gym_wrapper.py           # OpenAI Gym integration
├── models/                       # SNN models
│   └── policy_snn.py            # Policy network definitions
├── training/                     # Training pipeline
│   └── trainer.py               # Training logic
├── utils/                        # Utilities
│   ├── logger.py                # Logging
│   └── visualizer.py            # Visualization tools
├── results/                      # Outputs
│   ├── metrics/                 # Training metrics
│   └── visualizations/          # Plots and 3D visualizations
├── main.py                       # Main script
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container configuration
└── ARCHITECTURE.md              # Technical documentation
```

### Data Flow
```mermaid
graph TD
    A[Environment] -->|State| B[SNN Policy]
    B -->|Action| C[Action Selection]
    C -->|Action| A
    A -->|Reward, Next State| D[Experience Buffer]
    D -->|Trajectory| E[Agent]
    E -->|Policy Update| B
    E -->|Metrics| F[Logger/Visualizer]
```

---

## 🧠 Algorithms

### Neural Network Models
- **Spiking Neural Networks (SNNs)**:
  - **Leaky Integrate-and-Fire (LIF)**: Models membrane potential dynamics with decay factor `β`.
  - **Alpha Neurons**: Uses alpha synapses for temporal dynamics.
  - **Synaptic Neurons**: Incorporates synaptic conductances for complex interactions.
  - **Spike Generation**: Threshold-based spiking mechanism.

### Reinforcement Learning
- **Policy Gradient Methods**:
  - **REINFORCE**: Updates policy parameters using trajectory returns.
  - **A2C**: Combines policy and value function learning with advantage estimation.
  - **Entropy Regularization**: Encourages exploration.
- **Experience Collection**: Stores state-action-reward trajectories.
- **Spike-based Backpropagation**: Adapts backpropagation for discrete spikes.

### Training Methods
- **Policy Optimization**: Gradient ascent on expected rewards.
- **Value Function Learning**: Estimates state values for A2C.
- **Experience Replay**: Buffers trajectories for stable training.

---

## 📜 Design Principles

1. **SOLID Principles**:
   - **Single Responsibility**: Each module (e.g., `trainer.py`, `visualizer.py`) has a single purpose.
   - **Open/Closed**: Extensible for new neuron types and algorithms.
   - **Liskov Substitution**: Components (e.g., policies, action strategies) are interchangeable.
   - **Interface Segregation**: Focused interfaces for specific tasks.
   - **Dependency Inversion**: Abstracts dependencies for flexibility.

2. **Clean Architecture**:
   - Independent layers (agents, models, environments).
   - Inward dependency flow for modularity.
   - Isolated business rules (RL logic).
   - Framework-agnostic design.

3. **Modularity**:
   - Loose coupling between modules.
   - High cohesion within components.
   - Reusable and testable code.

---

## 🚀 System Features

1. **Extensibility**:
   - Add new environments via `environments/`.
   - Implement custom policies in `models/`.
   - Support new RL algorithms in `agents/`.

2. **Maintainability**:
   - Clear directory structure and documentation.
   - Comprehensive logging and visualization.
   - Unit testing support.

3. **Scalability**:
   - Multi-environment parallel training.
   - GPU-accelerated computation.
   - Resource-efficient design.

4. **Observability**:
   - Real-time logging of training metrics.
   - 3D visualizations of spike activity.
   - Detailed performance analysis.

---

## 🔍 Detailed Implementations

### Leaky Integrate-and-Fire (LIF) Neuron
```python
class LIFNeuron:
    def __init__(self, beta=0.95, threshold=1.0, reset=0.0):
        self.beta = beta
        self.threshold = threshold
        self.reset = reset
        self.mem = 0.0

    def forward(self, input_current):
        self.mem = self.beta * self.mem + input_current
        spike = 1 if self.mem > self.threshold else 0
        if spike:
            self.mem = self.reset
        return spike, self.mem
```

### REINFORCE Algorithm
```python
class REINFORCE:
    def update(self, trajectory, policy, optimizer):
        returns = compute_discounted_returns(trajectory.rewards, gamma=0.99)
        log_probs = trajectory.log_probs
        loss = -sum(log_prob * return_ for log_prob, return_ in zip(log_probs, returns))
        loss += entropy_regularization(policy, beta=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Factory Pattern
```python
class SNNFactory:
    @staticmethod
    def create_policy(policy_type, config):
        if policy_type == "LIF":
            return LIFPolicy(config["hidden_size"], config["num_steps"])
        elif policy_type == "Alpha":
            return AlphaPolicy(config["hidden_size"], config["num_steps"])
        elif policy_type == "Synaptic":
            return SynapticPolicy(config["hidden_size"], config["num_steps"])
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
```

### Strategy Pattern
```python
class ActionStrategy:
    def select_action(self, policy_output):
        pass

class SoftmaxStrategy(ActionStrategy):
    def select_action(self, policy_output):
        probs = torch.softmax(policy_output, dim=-1)
        return torch.multinomial(probs, 1).item()
```

---

## 📊 System Diagrams

### Training Pipeline
```mermaid
graph TD
    A[Environment] -->|State| B[SNN Policy]
    B -->|Action Prob| C[Action Selection]
    C -->|Action| A
    A -->|Reward, Next State| D[Experience Buffer]
    D -->|Trajectory| E[Agent]
    E -->|Policy Update| B
    E -->|Metrics| F[Logger/Visualizer]
```

### Neural Architecture
```mermaid
graph LR
    A[Input Layer<br>State (e.g., 4D)] --> B[Hidden Layer 1<br>LIF Neurons]
    B --> C[Hidden Layer 2<br>LIF Neurons]
    C --> D[Output Layer<br>Action Probabilities]
    E[Membrane Potential] --> B & C & D
    F[Spike Generation] --> B & C & D
```

### Data Processing Flow
```mermaid
graph LR
    A[Environment] -->|State| B[SNN Policy]
    B -->|Action| C[Action Selection]
    C -->|Action| A
    A -->|Reward, Next State| D[Experience Buffer]
    D -->|Trajectory| E[Policy Update]
    E --> B
    E -->|Metrics| F[Logger/Visualizer]
```

---