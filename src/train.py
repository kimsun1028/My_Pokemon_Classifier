"""Training script for Pokemon classifier models"""

# ============================================================================
# ⚙️ CONFIGURATION - 여기서 훈련 파라미터를 설정하세요
# ============================================================================
CONFIG = {
    'model': 'all',              # 훈련할 모델: 'alexnet', 'vggnet', 'googlenet', 'resnet', 'all'
    'epochs': 2,                # 에포크 수
    'batch_size': 32,            # 배치 크기
    'learning_rate': 1e-3,       # 학습률
    'weight_decay': 1e-5,        # 가중치 감쇠
    'device': 'auto',            # 디바이스: 'auto', 'cpu', 'gpu'
}
# ============================================================================

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

# Import analysis functions
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


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


def analyze_training_results(results_dir='results'):
    """Analyze and visualize training results"""
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️  matplotlib/pandas not available. Skipping visualization.")
        return
    
    print(f"\n{'='*60}")
    print("Analyzing training results...")
    print(f"{'='*60}\n")
    
    results_dir = Path(results_dir)
    models = ['alexnet', 'vggnet', 'googlenet', 'resnet']
    histories = {}
    
    # Load histories
    for model in models:
        history_file = results_dir / f'{model}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                histories[model] = json.load(f)
    
    if not histories:
        print("❌ No training results found.")
        return
    
    # 최종 성능 지표
    performance_data = []
    for model in histories:
        history = histories[model]
        performance_data.append({
            'Model': model.upper(),
            'Train Accuracy': history['train_accs'][-1],
            'Val Accuracy': history['val_accs'][-1],
            'Train Loss': history['train_losses'][-1],
            'Val Loss': history['val_losses'][-1]
        })
    
    df_performance = pd.DataFrame(performance_data)
    
    print("Model Performance Comparison")
    print("-" * 70)
    print(df_performance.to_string(index=False))
    print()
    
    # Best performing model
    best_model = df_performance.loc[df_performance['Val Accuracy'].idxmax()]
    print(f"🏆 Best Model: {best_model['Model']}")
    print(f"   Validation Accuracy: {best_model['Val Accuracy']:.4f}")
    print(f"   Validation Loss: {best_model['Val Loss']:.4f}\n")
    
    # 그래프 생성
    try:
        # 정확도 곡선
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pokemon Classification Model Accuracy Curves', fontsize=16, fontweight='bold')
        
        models_list = list(histories.keys())
        colors = {'train': '#2E86AB', 'val': '#A23B72'}
        axes = axes.flatten()
        
        for idx, model in enumerate(models_list):
            history = histories[model]
            epochs = range(1, len(history['train_accs']) + 1)
            
            ax = axes[idx]
            ax.plot(epochs, history['train_accs'], marker='o', label='Train Accuracy', 
                    color=colors['train'], linewidth=2, markersize=4)
            ax.plot(epochs, history['val_accs'], marker='s', label='Validation Accuracy', 
                    color=colors['val'], linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{model.upper()} - Accuracy', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        accuracy_path = results_dir / 'accuracy_curves.png'
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy curves saved: {accuracy_path}")
        plt.close()
        
        # 손실값 곡선
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pokemon Classification Model Loss Curves', fontsize=16, fontweight='bold')
        
        colors = {'train': '#F18F01', 'val': '#C73E1D'}
        axes = axes.flatten()
        
        for idx, model in enumerate(models_list):
            history = histories[model]
            epochs = range(1, len(history['train_losses']) + 1)
            
            ax = axes[idx]
            ax.plot(epochs, history['train_losses'], marker='o', label='Train Loss', 
                    color=colors['train'], linewidth=2, markersize=4)
            ax.plot(epochs, history['val_losses'], marker='s', label='Validation Loss', 
                    color=colors['val'], linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (Cross Entropy)')
            ax.set_title(f'{model.upper()} - Loss', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_path = results_dir / 'loss_curves.png'
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curves saved: {loss_path}")
        plt.close()
        
        # 모델 비교 그래프
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(df_performance))
        width = 0.35
        
        ax1.bar(x - width/2, df_performance['Train Accuracy'], width, label='Train Accuracy', color='#2E86AB')
        ax1.bar(x + width/2, df_performance['Val Accuracy'], width, label='Validation Accuracy', color='#A23B72')
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_performance['Model'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])
        
        ax2.bar(x - width/2, df_performance['Train Loss'], width, label='Train Loss', color='#F18F01')
        ax2.bar(x + width/2, df_performance['Val Loss'], width, label='Validation Loss', color='#C73E1D')
        
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_performance['Model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        comparison_path = results_dir / 'comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison graphs saved: {comparison_path}")
        plt.close()
        
        # 통합 정확도 곡선
        plt.figure(figsize=(12, 6))
        
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, model in enumerate(models_list):
            history = histories[model]
            epochs = range(1, len(history['val_accs']) + 1)
            plt.plot(epochs, history['val_accs'], marker='o', label=model.upper(), 
                    color=colors_list[idx], linewidth=2, markersize=5)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('All Models Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        combined_path = results_dir / 'combined_accuracy.png'
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined accuracy curves saved: {combined_path}")
        plt.close()
        
        print(f"\n✓ Analysis complete!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error during graph generation: {e}")
        print("   Analysis completed but visualization failed.")


def main(args):
    """Main training function"""
    
    print(f"\n{'='*60}")
    print(f"⚙️  설정 정보")
    print(f"{'='*60}")
    print(f"  모델: {args.model}")
    print(f"  에포크: {args.epochs}")
    print(f"  배치 크기: {args.batch_size}")
    print(f"  학습률: {args.lr}")
    print(f"{'='*60}\n")
    
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
    
    # Determine which models to train
    if args.model.lower() == 'all':
        models_to_train = ['alexnet', 'vggnet', 'googlenet', 'resnet']
        print(f"\nTraining all models: {', '.join(models_to_train)}")
    else:
        models_to_train = [args.model]
    
    # Train each model
    for model_name in models_to_train:
        # Get model
        print(f"\n{'='*60}")
        print(f"Initializing {model_name.upper()}...")
        print(f"{'='*60}")
        model = get_model(model_name, num_classes)
        print(model)
        
        # Train
        trainer = Trainer(
            model=model,
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )
        
        trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        # Save training history
        history = trainer.get_training_history()
        history_path = f'results/{model_name}_history.json'
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    print(f"\n{'='*60}")
    print("All training completed!")
    print(f"{'='*60}")
    
    # Analyze results
    analyze_training_results()


if __name__ == '__main__':
    # 항상 CONFIG 설정을 우선하여 사용합니다
    class Config:
        pass
    
    args = Config()
    args.model = CONFIG['model']
    args.epochs = CONFIG['epochs']
    args.batch_size = CONFIG['batch_size']
    args.lr = CONFIG['learning_rate']
    args.weight_decay = CONFIG['weight_decay']
    args.device = CONFIG['device']
    
    main(args)
