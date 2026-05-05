"""AlexNet Classifier for Pokemon classification"""

import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetClassifier(nn.Module):
    """AlexNet-based Pokemon classifier using pretrained weights"""
    
    def __init__(self, num_classes=150, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of Pokemon classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(AlexNetClassifier, self).__init__()
        
        # Load pretrained AlexNet
        self.backbone = models.alexnet(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)
        
        self.model_name = 'AlexNet'
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'name': self.model_name,
            'backbone': 'AlexNet',
            'pretrained': True,
            'framework': 'PyTorch'
        }


if __name__ == '__main__':
    # Test the model
    model = AlexNetClassifier(num_classes=150, pretrained=True)
    print(f"Model: {model.get_config()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
