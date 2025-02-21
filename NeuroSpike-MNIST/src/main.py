import torch
import snntorch as snn
from torchvision import datasets, transforms
from data_handler.data_loader import SNNDataLoader
from models.snn_model import SNNModel
from training.trainer import SNNTrainer
from utils.config import Config
from utils.visualization import SpikeVisualizer
import argparse
from inference import SNNInference
import os
import matplotlib.pyplot as plt

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])
    
    train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/', train=False, transform=transform)
    
    return train_dataset, test_dataset

def create_inference_samples(num_samples=10):
    """Create random inference samples from MNIST test dataset"""
    _, test_dataset = load_mnist()
    
    # Create samples directory if it doesn't exist
    samples_dir = 'inference_samples'
    os.makedirs(samples_dir, exist_ok=True)
    
    # Randomly select samples
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    sample_paths = []
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        # Convert tensor back to PIL Image
        image = transforms.ToPILImage()(image.view(28, 28))
        # Save image
        image_path = os.path.join(samples_dir, f'sample_{i}_label_{label}.png')
        image.save(image_path)
        sample_paths.append((image_path, label))
    
    return sample_paths

def visualize_training_metrics(trainer_metrics):
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(trainer_metrics['train_loss'], label='Training Loss')
    plt.plot(trainer_metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(trainer_metrics['train_acc'], label='Training Accuracy')
    plt.plot(trainer_metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_metrics.png')
    plt.close()

def visualize_inference(image, prediction, true_label=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    title = f'Predicted: {prediction}'
    if true_label is not None:
        title += f' (True: {true_label})'
    plt.title(title)
    plt.axis('off')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/inference_result.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='NeuroSpike-MNIST Training/Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'create_samples'],
                      help='Mode: train new model, use pretrained model, or create inference samples')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to pretrained model for inference')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of inference samples to create')
    args = parser.parse_args()

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    if args.mode == 'create_samples':
        sample_paths = create_inference_samples(args.num_samples)
        print("Created inference samples:")
        for path, label in sample_paths:
            print(f"Image: {path}, True Label: {label}")
        return

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
        
        # Train model and get metrics
        metrics = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=config.training.epochs
        )
        
        # Visualize training metrics
        visualize_training_metrics(metrics)
        print("Training visualization saved to plots/training_metrics.png")
    
    else:  # Inference mode
        if not args.image_path:
            raise ValueError("Please provide an image path for inference mode")
            
        # Initialize inference
        inferencer = SNNInference(args.model_path, 'config.yaml')
        
        # Load and predict
        from PIL import Image
        image = Image.open(args.image_path).convert('L')
        prediction = inferencer.predict(image)
        
        # Extract true label from filename if available
        true_label = None
        if 'label_' in args.image_path:
            try:
                true_label = int(args.image_path.split('label_')[1].split('.')[0])
            except:
                pass
        
        # Visualize inference result
        visualize_inference(image, prediction, true_label)
        print(f"Predicted digit: {prediction}")
        if true_label is not None:
            print(f"True digit: {true_label}")
        print("Inference visualization saved to plots/inference_result.png")

if __name__ == "__main__":
    main()