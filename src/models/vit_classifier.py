"""GoogleNet (Inception) Classifier for Pokemon classification"""

import torch
import torch.nn as nn
import torchvision.models as models


class GoogleNetClassifier(nn.Module):
    """GoogleNet (Inception) based Pokemon classifier using pretrained weights"""
    
    def __init__(self, num_classes=150, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of Pokemon classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(GoogleNetClassifier, self).__init__()
        
        # Load pretrained GoogleNet (Inception v1)
        self.backbone = models.googlenet(pretrained=pretrained, aux_logits=False)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.model_name = 'GoogleNet'
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'name': self.model_name,
            'backbone': 'GoogleNet (Inception)',
            'pretrained': True,
            'framework': 'PyTorch'
        }


if __name__ == '__main__':
    # Test the model
    model = GoogleNetClassifier(num_classes=150, pretrained=True)
    print(f"Model: {model.get_config()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
