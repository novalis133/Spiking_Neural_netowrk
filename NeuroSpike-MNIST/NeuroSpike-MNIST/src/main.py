import torch
import snntorch as snn
from torchvision import datasets, transforms
from data.data_loader import SNNDataLoader
from models.snn_model import SNNModel
from training.trainer import SNNTrainer
from utils.config import Config
from utils.visualization import SpikeVisualizer
import argparse
from inference import SNNInference

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])
    
    train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/', train=False, transform=transform)
    
    return train_dataset, test_dataset

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='NeuroSpike-MNIST Training/Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                      help='Mode: train new model or use pretrained model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to pretrained model for inference')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    args = parser.parse_args()

    # Load configuration
    config = Config('config.yaml')

    if args.mode == 'train':
        # Existing training code
        train_dataset, test_dataset = load_mnist()
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.training.batch_size, 
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False
        )
        
        # Initialize model
        model = SNNModel(
            input_size=784,
            hidden_size=config.model.hidden_size,
            output_size=10,
            num_steps=config.model.num_steps,
            beta=config.model.beta
        )
        
        # Initialize optimizer and trainer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        trainer = SNNTrainer(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss()
        )
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=config.training.epochs
        )
    
    else:  # Inference mode
        if not args.image_path:
            raise ValueError("Please provide an image path for inference mode")
            
        # Initialize inference
        inferencer = SNNInference(args.model_path, 'config.yaml')
        
        # Load and predict
        from PIL import Image
        image = Image.open(args.image_path).convert('L')
        prediction = inferencer.predict(image)
        print(f"Predicted digit: {prediction}")

if __name__ == "__main__":
    main()