import gym
import numpy as np
from typing import Dict, Any

class MultiEnvironmentManager:
    SUPPORTED_ENVS = {
        'cartpole': 'CartPole-v1',
        'acrobot': 'Acrobot-v1',
        'mountaincar': 'MountainCar-v0',
        'lunarlander': 'LunarLander-v2'
    }
    
    def __init__(self):
        self.envs: Dict[str, Any] = {}
        
    def create_env(self, env_name: str):
        if env_name not in self.SUPPORTED_ENVS:
            raise ValueError(f"Environment {env_name} not supported")
            
        if env_name not in self.envs:
            self.envs[env_name] = gym.make(self.SUPPORTED_ENVS[env_name])
            
        return self.envs[env_name]
    
    def get_env_info(self, env_name: str):
        env = self.create_env(env_name)
        return {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'max_steps': env._max_episode_steps
        }
    
    def close_all(self):
        for env in self.envs.values():
            env.close()