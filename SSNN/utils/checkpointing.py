import torch
from pathlib import Path
import json
from datetime import datetime

class ModelCheckpointer:
    def __init__(self, save_dir: str = 'checkpoints'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, metrics, model_name, bp_method):
        checkpoint_dir = self.save_dir / model_name / bp_method
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': metrics,
            'model_name': model_name,
            'bp_method': bp_method
        }
        
        metadata_path = checkpoint_dir / f"metadata_epoch_{epoch}_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def load_checkpoint(self, model, optimizer, model_name, bp_method, epoch=None):
        checkpoint_dir = self.save_dir / model_name / bp_method
        
        if epoch is None:
            # Load latest checkpoint
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
        else:
            # Load specific epoch
            checkpoints = list(checkpoint_dir.glob(f"checkpoint_epoch_{epoch}_*.pt"))
            if not checkpoints:
                return None
            checkpoint_path = checkpoints[0]
            
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']