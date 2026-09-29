import torch
import torch.nn as nn
class SiameseChangeNet(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU());self.head=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,1,1))
    def forward(self,a,b): return nn.functional.interpolate(self.head((self.encoder(a)-self.encoder(b)).abs()),size=a.shape[-2:],mode='bilinear',align_corners=False)
