import torch.nn as nn, torch.nn.functional as F
class DiceBCELoss(nn.Module):
    def forward(self,logits,target):
        bce=F.binary_cross_entropy_with_logits(logits,target); p=logits.sigmoid(); inter=(p*target).sum((1,2,3)); den=p.sum((1,2,3))+target.sum((1,2,3)); dice=((2*inter+1e-6)/(den+1e-6)).mean(); return bce+(1-dice)
