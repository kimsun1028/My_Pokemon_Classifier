"""ResNet50 Classifier for Pokemon classification"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """ResNet50-based Pokemon classifier using pretrained weights"""
    
    def __init__(self, num_classes=150, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of Pokemon classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.model_name = 'ResNet50'
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'name': self.model_name,
            'backbone': 'ResNet50',
            'pretrained': True,
            'framework': 'PyTorch'
        }


if __name__ == '__main__':
    # Test the model
    model = ResNetClassifier(num_classes=150, pretrained=True)
    print(f"Model: {model.get_config()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
