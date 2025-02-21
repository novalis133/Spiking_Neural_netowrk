import torch
import numpy as np
from collections import deque

class SNNAgent:
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=config.memory_size)
        
    def select_action(self, state):
        return self.policy.act(state, self.config.temperature)
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def compute_returns(self, rewards, gamma):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns