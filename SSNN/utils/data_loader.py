import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import utils

class DataManager:
    def __init__(self, config, data_path="./data"):
        self.config = config
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

    def get_mnist(self):
        train_dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.MNIST(
            self.data_path, train=False, download=True, transform=self.transform
        )
        
        if self.config.subset_size > 1:
            utils.data_subset(train_dataset, self.config.subset_size)
            utils.data_subset(test_dataset, self.config.subset_size)
        
        return self._create_loaders(train_dataset, test_dataset)

    def get_fashion_mnist(self):
        train_dataset = datasets.FashionMNIST(
            self.data_path, train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.FashionMNIST(
            self.data_path, train=False, download=True, transform=self.transform
        )
        
        if self.config.subset_size > 1:
            utils.data_subset(train_dataset, self.config.subset_size)
            utils.data_subset(test_dataset, self.config.subset_size)
        
        return self._create_loaders(train_dataset, test_dataset)

    def _create_loaders(self, train_dataset, test_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader