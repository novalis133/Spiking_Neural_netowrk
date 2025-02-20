import torch
from pathlib import Path
import json
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, episode, metrics, experiment_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'episode': episode,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        path = self.save_dir / f"{experiment_name}_episode_{episode}_{timestamp}.pt"
        torch.save(checkpoint, path)
        
        # Save metadata
        metadata = {
            'episode': episode,
            'metrics': metrics,
            'timestamp': timestamp
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def load_latest_checkpoint(self, model, optimizer, experiment_name):
        checkpoints = list(self.save_dir.glob(f"{experiment_name}*.pt"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return checkpoint['episode'], checkpoint['metrics']