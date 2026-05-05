"""Evaluation script for Pokemon classifier models"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

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


class Evaluator:
    """Model Evaluator class"""
    
    def __init__(self, model, model_name, device):
        """Initialize evaluator"""
        self.device = device
        self.model = model.to(self.device)
        self.model_name = model_name
        self.model.eval()
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: Checkpoint {filepath} not found")
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        all_preds = []
        all_labels = []
        
        print("Evaluating...")
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Evaluating')
            for images, labels in progress_bar:
                images = images.to(self.device, non_blocking=self.device.type == 'cuda')
                labels = labels.to(self.device, non_blocking=self.device.type == 'cuda')
                
                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        return metrics, all_labels, all_preds
    
    def save_results(self, metrics, output_path='results/eval_results.json'):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'model': self.model_name,
            'metrics': metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {output_path}")


def get_model(model_name, num_classes):
    """Get model instance"""
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'alexnet':
        return AlexNetClassifier(num_classes=num_classes, pretrained=False)
    elif model_name_lower == 'vggnet':
        return VGGNetClassifier(num_classes=num_classes, model_name='vgg16', pretrained=False)
    elif model_name_lower == 'googlenet':
        return GoogleNetClassifier(num_classes=num_classes, pretrained=False)
    elif model_name_lower == 'resnet':
        return ResNetClassifier(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(args):
    """Main evaluation function"""
    
    # Create dataloaders
    print("Loading data...")
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    dataloaders = create_dataloaders(
        data_dir='data/raw',
        batch_size=args.batch_size,
        augment=False,
        pin_memory=device.type == 'cuda'
    )
    
    test_loader = dataloaders['test']
    num_classes = dataloaders['num_classes']
    class_names = dataloaders['class_names']
    
    print(f"Number of classes: {num_classes}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get model
    print(f"\nInitializing {args.model}...")
    model = get_model(args.model, num_classes)
    
    # Evaluator
    evaluator = Evaluator(model=model, model_name=args.model, device=device)
    
    # Load checkpoint
    checkpoint_path = f'results/{args.model}_best.pth'
    evaluator.load_checkpoint(checkpoint_path)
    
    # Evaluate
    metrics, all_labels, all_preds = evaluator.evaluate(test_loader)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {args.model}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*50}")
    
    # Save results
    results_path = f'results/{args.model}_metrics.json'
    evaluator.save_results(metrics, results_path)
    
    # Save detailed classification report
    if args.save_report:
        report_path = f'results/{args.model}_classification_report.txt'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(classification_report(all_labels, all_preds, target_names=class_names))
        print(f"Classification report saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pokemon classifier')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['alexnet', 'vggnet', 'googlenet', 'resnet'],
                        help='Model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_report', action='store_true', help='Save classification report')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='Device to use')
    
    args = parser.parse_args()
    main(args)
