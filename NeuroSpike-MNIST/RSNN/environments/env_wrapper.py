import gym
import numpy as np

class RLEnvironment:
    def __init__(self, env_name='CartPole-v1'):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        return self.env.reset()[0]
        
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done or truncated, info
        
    def close(self):
        self.env.close()