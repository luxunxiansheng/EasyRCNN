import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, 
                in_channels:int,
                out_channels:int, 
                kernel_size:int,
                stride:int=1,
                relu:bool=True, 
                same_padding:bool=False, 
                bn:bool=False):

        super().__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, 
                in_features:int,
                out_features:int, 
                relu:bool=True):

        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x:torch.Tensor):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)