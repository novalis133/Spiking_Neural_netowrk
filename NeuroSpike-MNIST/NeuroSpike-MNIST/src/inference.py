import torch
from models.snn_model import SNNModel
from utils.config import Config
from torchvision import transforms
import numpy as np

class SNNInference:
    def __init__(self, model_path, config_path='config.yaml'):
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SNNModel(
            input_size=784,
            hidden_size=self.config.model.hidden_size,
            output_size=10,
            num_steps=self.config.model.num_steps,
            beta=self.config.model.beta
        ).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
    
    def predict(self, image):
        """
        Predict digit from image
        image: PIL Image or numpy array of shape (28, 28)
        """
        # Prepare input
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            spike_output = self.model(image)
            spike_output = spike_output.mean(dim=0)
            prediction = spike_output.argmax(dim=1)
        
        return prediction.item()

# Usage example
if __name__ == "__main__":
    # Initialize inference
    inferencer = SNNInference('checkpoints/best_model.pth')
    
    # Load and predict single image
    from PIL import Image
    image = Image.open('path/to/digit.png').convert('L')
    prediction = inferencer.predict(image)
    print(f"Predicted digit: {prediction}")