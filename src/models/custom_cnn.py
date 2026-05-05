"""VGGNet Classifier for Pokemon classification"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGNetClassifier(nn.Module):
    """VGGNet-based Pokemon classifier using pretrained weights"""
    
    def __init__(self, num_classes=150, model_name='vgg16', pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of Pokemon classes
            model_name: VGG variant (vgg11, vgg13, vgg16, vgg19)
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(VGGNetClassifier, self).__init__()
        
        # Load pretrained VGG
        if model_name == 'vgg11':
            self.backbone = models.vgg11(pretrained=pretrained)
        elif model_name == 'vgg13':
            self.backbone = models.vgg13(pretrained=pretrained)
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg19':
            self.backbone = models.vgg19(pretrained=pretrained)
        else:
            self.backbone = models.vgg16(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)
        
        self.model_name = f'VGG-{model_name.upper()}'
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'name': self.model_name,
            'backbone': 'VGGNet',
            'pretrained': True,
            'framework': 'PyTorch'
        }


if __name__ == '__main__':
    # Test the model
    model = VGGNetClassifier(num_classes=150, model_name='vgg16')
    print(f"Model: {model.get_config()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
