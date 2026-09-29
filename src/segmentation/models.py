import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,a,b): super().__init__(); self.m=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
    def forward(self,x): return self.m(x)
class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.p=nn.MaxPool2d(2); self.e1=DoubleConv(3,64); self.e2=DoubleConv(64,128); self.e3=DoubleConv(128,256); self.e4=DoubleConv(256,512); self.b=DoubleConv(512,1024); self.u4=nn.ConvTranspose2d(1024,512,2,2); self.c4=DoubleConv(1024,512); self.u3=nn.ConvTranspose2d(512,256,2,2); self.c3=DoubleConv(512,256); self.u2=nn.ConvTranspose2d(256,128,2,2); self.c2=DoubleConv(256,128); self.u1=nn.ConvTranspose2d(128,64,2,2); self.c1=DoubleConv(128,64); self.out=nn.Conv2d(64,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.p(a)); c=self.e3(self.p(b)); d=self.e4(self.p(c)); z=self.b(self.p(d)); z=self.c4(__import__('torch').cat([self.u4(z),d],1)); z=self.c3(__import__('torch').cat([self.u3(z),c],1)); z=self.c2(__import__('torch').cat([self.u2(z),b],1)); z=self.c1(__import__('torch').cat([self.u1(z),a],1)); return self.out(z)
def build_model(name='unet'):
    if name=='unet': return UNet()
    if name=='segformer':
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0',num_labels=2,ignore_mismatched_sizes=True)
    raise ValueError(name)
