"""Data loader and preprocessing for Pokemon classifier"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')


class PokemonDataset(Dataset):
    """Custom Dataset for Pokemon images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            transform: Optional transforms to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_data_transforms(input_size=224, augment=True):
    """
    Create train and validation transforms
    
    Args:
        input_size: Input image size (default: 224)
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {'train': train_transform, 'val': val_transform}


def load_pokemon_data(data_dir='data/raw', test_size=0.2, val_size=0.1):
    """
    Load Pokemon dataset from directory
    
    Args:
        data_dir: Directory containing pokemon images organized by class
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
    
    Returns:
        Dictionary with train/val/test splits and class info
    """
    image_paths = []
    labels = []
    class_names = []
    class_to_idx = {}
    
    data_path = Path(data_dir)
    
    # Check if PokemonData folder exists
    pokemon_data_path = data_path / 'PokemonData'
    if pokemon_data_path.exists():
        data_path = pokemon_data_path
    
    # Collect all images organized by folders (each folder = class)
    for class_idx, class_dir in enumerate(sorted(data_path.iterdir())):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            class_to_idx[class_name] = class_idx
            
            # Get all image files in this class directory
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in image_extensions:
                    image_paths.append(str(image_path))
                    labels.append(class_idx)
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(test_size + val_size), random_state=42
    )
    
    val_test_split = val_size / (test_size + val_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=val_test_split, random_state=42
    )
    
    return {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels},
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': len(class_names)
    }


def create_dataloaders(data_dir='data/raw', batch_size=32, num_workers=4, augment=True, pin_memory=None):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Directory containing pokemon images
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary with dataloaders and dataset info
    """
    if os.name == 'nt':
        num_workers = 0

    # Load data
    data_info = load_pokemon_data(data_dir)
    
    # Get transforms
    transforms_dict = get_data_transforms(augment=augment)
    
    # Create datasets
    train_dataset = PokemonDataset(
        data_info['train']['paths'],
        data_info['train']['labels'],
        transform=transforms_dict['train']
    )
    
    val_dataset = PokemonDataset(
        data_info['val']['paths'],
        data_info['val']['labels'],
        transform=transforms_dict['val']
    )
    
    test_dataset = PokemonDataset(
        data_info['test']['paths'],
        data_info['test']['labels'],
        transform=transforms_dict['val']
    )
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': data_info['class_names'],
        'num_classes': data_info['num_classes']
    }


if __name__ == '__main__':
    # Example usage
    print("Loading Pokemon data...")
    data_info = load_pokemon_data()
    print(f"Number of classes: {data_info['num_classes']}")
    print(f"Training samples: {len(data_info['train']['labels'])}")
    print(f"Validation samples: {len(data_info['val']['labels'])}")
    print(f"Test samples: {len(data_info['test']['labels'])}")
