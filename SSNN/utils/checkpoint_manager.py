import shutil
from pathlib import Path
import json
import torch
from datetime import datetime

class CheckpointManager:
    def __init__(self, base_dir='checkpoints'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def list_checkpoints(self, experiment_name):
        exp_dir = self.base_dir / experiment_name
        checkpoints = []
        
        if exp_dir.exists():
            for cp_file in exp_dir.glob('*.pt'):
                meta_file = cp_file.with_suffix('.json')
                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        'path': cp_file,
                        'metadata': metadata
                    })
        
        return checkpoints
    
    def clean_old_checkpoints(self, experiment_name, keep_last_n=5):
        checkpoints = self.list_checkpoints(experiment_name)
        checkpoints.sort(key=lambda x: x['metadata']['timestamp'])
        
        if len(checkpoints) > keep_last_n:
            for cp in checkpoints[:-keep_last_n]:
                cp['path'].unlink()
                cp['path'].with_suffix('.json').unlink()
                
    def export_checkpoint(self, source_path, target_dir):
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True)
        
        source_path = Path(source_path)
        shutil.copy2(source_path, target_dir / source_path.name)
        shutil.copy2(source_path.with_suffix('.json'), 
                    target_dir / source_path.with_suffix('.json').name)