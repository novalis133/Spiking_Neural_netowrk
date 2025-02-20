import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger('RSNN')
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def log_metrics(self, episode, metrics):
        self.logger.info(f"Episode {episode}: {json.dumps(metrics)}")
        
    def plot_learning_curve(self, rewards, window_size=100):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'))
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.log_dir / 'learning_curve.png')
        plt.close()