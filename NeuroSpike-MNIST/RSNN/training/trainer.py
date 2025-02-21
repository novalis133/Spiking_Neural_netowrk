import torch
import numpy as np
from tqdm import tqdm
import snntorch.functional as SF
from torch.utils.data import DataLoader
from ..utils.dataset import RLDataset

class RSNNTrainer:
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.loss_fn = SF.mse_count_loss()
        self.reg_fn = SF.l1_rate_sparsity()
        
    def train_episode(self):
        state = self.env.reset()
        episode_reward = 0
        saved_log_probs = []
        rewards = []
        
        for t in range(self.config.max_steps):
            action, log_prob = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            saved_log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            
            self.agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
                
        returns = self.agent.compute_returns(rewards, self.config.gamma)
        policy_loss = []
        
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        return torch.stack(policy_loss).sum(), episode_reward