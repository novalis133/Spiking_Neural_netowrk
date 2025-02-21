import torch
from configs.config import NetworkConfig, RLConfig, TrainingConfig
from configs.experiments import EXPERIMENTS
from models.policy_snn import PolicySNN
from agents.snn_agent import SNNAgent
from environments.env_wrapper import RLEnvironment
from training.trainer import RSNNTrainer
from utils.logger import ExperimentLogger
from utils.visualizer import Visualizer
from utils.checkpointing import CheckpointManager
from utils.metrics import PerformanceMetrics

def run_experiment(experiment_config):
    # Initialize components
    env = RLEnvironment(experiment_config.env_name)
    network = PolicySNN(NetworkConfig(**experiment_config.network_params))
    agent = SNNAgent(network, RLConfig())
    trainer = RSNNTrainer(agent, env, TrainingConfig(**experiment_config.training_params))
    
    # Initialize tracking tools
    logger = ExperimentLogger(f'logs/{experiment_config.name}')
    visualizer = Visualizer(f'visualizations/{experiment_config.name}')
    checkpoint_mgr = CheckpointManager(f'checkpoints/{experiment_config.name}')
    metrics = PerformanceMetrics()
    
    # Training loop
    for episode in range(experiment_config.training_params['num_episodes']):
        loss, reward, info = trainer.train_episode()
        metrics.update(reward, info['length'], loss.item(), info['spike_rate'])
        
        if episode % 100 == 0:
            current_metrics = metrics.get_metrics()
            logger.log_metrics(episode, current_metrics)
            checkpoint_mgr.save_checkpoint(network, trainer.optimizer, episode, current_metrics, experiment_config.name)
            visualizer.plot_training_metrics(current_metrics)

def main():
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\nRunning experiment: {exp_name}")
        run_experiment(exp_config)

if __name__ == "__main__":
    main()