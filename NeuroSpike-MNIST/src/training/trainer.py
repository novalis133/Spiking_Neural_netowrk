import torch
from tqdm import tqdm
import snntorch as snn

class SNNTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            spike_output = self.model(data)
            
            # Average across time steps for classification
            spike_output = spike_output.mean(dim=0)  # Average across time dimension
            
            # Calculate loss
            loss = self.criterion(spike_output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                spike_output = self.model(data)
                
                # Average across time steps for classification
                spike_output = spike_output.mean(dim=0)  # Average across time dimension
                
                # Calculate loss
                loss = self.criterion(spike_output, target)
                total_loss += loss.item()
                
                # Get predictions
                pred = spike_output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = correct / len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs):
        best_accuracy = 0
        checkpoint_path = 'checkpoints/best_model.pth'
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, accuracy = self.validate(val_loader)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch
                }, checkpoint_path)