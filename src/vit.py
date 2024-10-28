import torchvision
from torch import nn

class VitBase16(nn.Module):
    def __init__(self, num_classes, device):
        super(VitBase16, self).__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)

    def forward(self, x):
        return self.vit(x)