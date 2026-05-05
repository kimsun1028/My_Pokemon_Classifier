"""Training script for Pokemon classifier models"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import create_dataloaders
from models import ResNetClassifier, AlexNetClassifier, VGGNetClassifier, GoogleNetClassifier

warnings.filterwarnings('ignore')


def resolve_device(device_choice):
    """Resolve the requested device."""
    if device_choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_choice == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('GPU requested, but CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device('cpu')


class Trainer:
    """Model Trainer class"""
    
    def __init__(self, model, model_name, num_classes, device, learning_rate=1e-3, weight_decay=1e-5):
        """Initialize trainer"""
        self.device = device
        self.model = model.to(self.device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False, dynamic_ncols=True)
        for images, labels in progress_bar:
            images = images.to(self.device, non_blocking=self.device.type == 'cuda')
            labels = labels.to(self.device, non_blocking=self.device.type == 'cuda')
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validating', leave=False, dynamic_ncols=True)
            for images, labels in progress_bar:
                images = images.to(self.device, non_blocking=self.device.type == 'cuda')
                labels = labels.to(self.device, non_blocking=self.device.type == 'cuda')
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50):
        """Train model for specified epochs"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\nStarting training for {self.model_name}...")
        overall_bar = tqdm(total=epochs, desc=f'Training {self.model_name}', dynamic_ncols=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            overall_bar.update(1)
            overall_bar.set_postfix({
                'epoch': f'{epoch+1}/{epochs}',
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.4f}'
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint(f'results/{self.model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        overall_bar.close()
        print(f"Training completed for {self.model_name}")
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def get_training_history(self):
        """Return training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }


def get_model(model_name, num_classes):
    """Get model instance"""
    model_name = model_name.lower()
    
    if model_name == 'alexnet':
        return AlexNetClassifier(num_classes=num_classes, pretrained=False)
    elif model_name == 'vggnet':
        return VGGNetClassifier(num_classes=num_classes, model_name='vgg16', pretrained=False)
    elif model_name == 'googlenet':
        return GoogleNetClassifier(num_classes=num_classes, pretrained=False)
    elif model_name == 'resnet':
        return ResNetClassifier(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(args):
    """Main training function"""
    
    # Create dataloaders
    print("Loading data...")
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    dataloaders = create_dataloaders(
        data_dir='data/raw',
        batch_size=args.batch_size,
        augment=True,
        pin_memory=device.type == 'cuda'
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    num_classes = dataloaders['num_classes']
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Get model
    print(f"\nInitializing {args.model}...")
    model = get_model(args.model, num_classes)
    print(model)
    
    # Train
    trainer = Trainer(
        model=model,
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    trainer.train(train_loader, val_loader, epochs=args.epochs)
    
    # Save training history
    history = trainer.get_training_history()
    history_path = f'results/{args.model}_history.json'
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pokemon classifier')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['alexnet', 'vggnet', 'googlenet', 'resnet'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='Device to use')
    
    args = parser.parse_args()
    main(args)
