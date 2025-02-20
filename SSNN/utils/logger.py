import logging
import wandb
from pathlib import Path

class ExperimentLogger:
    def __init__(self, experiment_name, model_name, bp_method):
        self.setup_file_logging(experiment_name, model_name, bp_method)
        self.setup_wandb(experiment_name, model_name, bp_method)
        
    def setup_file_logging(self, experiment_name, model_name, bp_method):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{model_name}_{bp_method}")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_dir / f"{experiment_name}_{model_name}_{bp_method}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def setup_wandb(self, experiment_name, model_name, bp_method):
        self.run = wandb.init(
            project="SSNN",
            name=f"{experiment_name}_{model_name}_{bp_method}",
            config={
                "model": model_name,
                "bp_method": bp_method,
                "experiment": experiment_name
            }
        )
        
    def log_metrics(self, metrics, step):
        self.logger.info(f"Step {step}: {metrics}")
        wandb.log(metrics, step=step)
        
    def close(self):
        wandb.finish()