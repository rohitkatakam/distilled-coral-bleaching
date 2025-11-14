"""Teacher model implementation using ResNet50 for coral bleaching classification."""

import torch
import torch.nn as nn
from torchvision import models


class TeacherModel(nn.Module):
    """
    Teacher model based on ResNet50 architecture.

    Uses pretrained ImageNet weights and replaces the final fully connected
    layer for binary classification (bleached vs healthy coral).

    Args:
        num_classes (int): Number of output classes. Default: 2 (bleached, healthy)
        pretrained (bool): Whether to load pretrained ImageNet weights. Default: True
        dropout (float, optional): Dropout probability for final layer. If None, no dropout.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=None):
        super(TeacherModel, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_p = dropout

        # Load ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)

        # Get the number of features from the original FC layer
        num_features = self.backbone.fc.in_features

        # Replace the final FC layer
        if dropout is not None and dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freeze all layers except the final classification layer.
        Useful for fine-tuning with limited data.
        """
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the final FC layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_trainable_params(self):
        """
        Get the number of trainable parameters.

        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        """
        Get the total number of parameters.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
