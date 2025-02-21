import torch
from snntorch import functional as SF
from pathlib import Path
import csv
from datetime import datetime
from tqdm import tqdm
from utils.checkpointing import ModelCheckpointer

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.optimizer_betas,
            eps=config.optimizer_eps
        )
        self.loss_fn = SF.ce_rate_loss()
        self.reg_fn = SF.l1_rate_sparsity()
        self.checkpointer = ModelCheckpointer()
        
    def train(self, train_loader, test_loader, bp_method):
        model_name = self.model.__class__.__name__
        bp_name = bp_method.__name__
        
        # Try to load checkpoint
        last_epoch, metrics = self.checkpointer.load_checkpoint(
            self.model, 
            self.optimizer,
            model_name,
            bp_name
        ) or (0, None)
        
        results_dir = Path(f'results/{model_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = results_dir / f"{bp_name}_{model_name}_Adamax_BS_{self.config.batch_size}.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["BP", "Model", "Epoch", "Train Loss", "Train Acc", "Test Acc"])
            
            for epoch in tqdm(range(last_epoch, self.config.epochs)):
                train_loss = bp_method(
                    self.model,
                    train_loader,
                    optimizer=self.optimizer,
                    criterion=self.loss_fn,
                    num_steps=self.config.num_steps,
                    time_var=False,
                    regularization=self.reg_fn,
                    device=self.device
                )
                
                train_acc = self._compute_accuracy(train_loader)
                test_acc = self._compute_accuracy(test_loader)
                
                writer.writerow([
                    bp_name, model_name, epoch,
                    train_loss.item(), train_acc * 100, test_acc * 100
                ])
                
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
                
                # Save checkpoint every N epochs
                if epoch % self.config.checkpoint_frequency == 0:
                    metrics = {
                        'train_loss': train_loss.item(),
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    }
                    self.checkpointer.save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        metrics,
                        model_name,
                        bp_name
                    )
    def _compute_accuracy(self, loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            self.model.eval()
            for data, targets in loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        self.model.train()
        return correct / total