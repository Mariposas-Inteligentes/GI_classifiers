import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtTiny, self).__init__()
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)  
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)  

    def forward(self, x):
        return self.model(x)
