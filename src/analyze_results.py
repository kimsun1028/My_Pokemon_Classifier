"""Analyze and visualize training results from all models"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set matplotlib to use default font for better English support
plt.rcParams['axes.unicode_minus'] = False


def load_histories(results_dir='results'):
    """Load training history from all models"""
    results_dir = Path(results_dir)
    models = ['alexnet', 'vggnet', 'googlenet', 'resnet']
    histories = {}
    
    print("Loading training histories...")
    for model in models:
        history_file = results_dir / f'{model}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                histories[model] = json.load(f)
            print(f"  ✓ {model.upper()} history loaded")
        else:
            print(f"  ✗ {model.upper()} history file not found")
    
    print(f"\nLoaded {len(histories)} model histories\n")
    return histories


def plot_accuracy_curves(histories):
    """Plot training and validation accuracy curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pokemon Classifier Model Accuracy Curves', fontsize=16, fontweight='bold')
    
    models = list(histories.keys())
    colors = {'train': '#2E86AB', 'val': '#A23B72'}
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
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
    plt.savefig('results/accuracy_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Accuracy curves saved: results/accuracy_curves.png")


def plot_loss_curves(histories):
    """Plot training and validation loss curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pokemon Classifier Model Loss Curves', fontsize=16, fontweight='bold')
    
    models = list(histories.keys())
    colors = {'train': '#F18F01', 'val': '#C73E1D'}
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        history = histories[model]
        epochs = range(1, len(history['train_losses']) + 1)
        
        ax = axes[idx]
        ax.plot(epochs, history['train_losses'], marker='o', label='Train Loss', 
                color=colors['train'], linewidth=2, markersize=4)
        ax.plot(epochs, history['val_losses'], marker='s', label='Validation Loss', 
                color=colors['val'], linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Cross Entropy Loss)')
        ax.set_title(f'{model.upper()} - Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/loss_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Loss curves saved: results/loss_curves.png")


def compare_models(histories):
    """Create comparison plots and summary table"""
    # Extract final performance metrics
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
    
    # Print results
    print("\n" + "="*80)
    print("Final Model Performance Comparison")
    print("="*80)
    print(df_performance.to_string(index=False))
    print()
    
    # Best performing model
    best_model = df_performance.loc[df_performance['Val Accuracy'].idxmax()]
    print(f"🏆 Best Performing Model: {best_model['Model']}")
    print(f"   Validation Accuracy: {best_model['Val Accuracy']:.4f}")
    print(f"   Validation Loss: {best_model['Val Loss']:.4f}")
    print("="*80 + "\n")
    
    # Comparison graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(df_performance))
    width = 0.35
    
    # Accuracy comparison
    ax1.bar(x - width/2, df_performance['Train Accuracy'], width, label='Train Accuracy', color='#2E86AB')
    ax1.bar(x + width/2, df_performance['Val Accuracy'], width, label='Validation Accuracy', color='#A23B72')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Final Accuracy Comparison by Model', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_performance['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Loss comparison
    ax2.bar(x - width/2, df_performance['Train Loss'], width, label='Train Loss', color='#F18F01')
    ax2.bar(x + width/2, df_performance['Val Loss'], width, label='Validation Loss', color='#C73E1D')
    
    ax2.set_ylabel('Loss')
    ax2.set_title('Final Loss Comparison by Model', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_performance['Model'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison graph saved: results/comparison.png")
    
    return df_performance


def plot_combined_accuracy(histories):
    """Plot all models' accuracy on single graph"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, model in enumerate(histories):
        history = histories[model]
        epochs = range(1, len(history['val_accs']) + 1)
        plt.plot(epochs, history['val_accs'], marker='o', label=model.upper(), 
                color=colors[idx], linewidth=2, markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Validation Accuracy Comparison - All Models', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/combined_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Combined accuracy curve saved: results/combined_accuracy.png")


def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("Pokemon Classifier Model Results Analysis")
    print("="*80 + "\n")
    
    # Load data
    histories = load_histories()
    
    if not histories:
        print("❌ No training results found. Please run train.py first.")
        return
    
    # Generate graphs
    print("Generating graphs...")
    plot_accuracy_curves(histories)
    plot_loss_curves(histories)
    plot_combined_accuracy(histories)
    
    # Compare models
    print()
    compare_models(histories)
    
    print("✓ Analysis complete!")
    print("  Generated graphs have been saved in the results/ folder.\n")


if __name__ == '__main__':
    main()
