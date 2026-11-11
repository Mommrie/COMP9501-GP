import math
import torch
from torch import nn


class Coefficients(nn.Module):
    def __init__(self,layers=1):
        super().__init__();self.logits=nn.Parameter(torch.zeros(layers,4))
        self.gain_logit=nn.Parameter(torch.tensor(math.log(1.5/2.0)))
    def weights(self):return self.logits.softmax(-1)
    def gain(self):return .5+3.5*self.gain_logit.sigmoid()
    def penalty(self):return (self.weights()-.25).square().sum(-1).mean()+torch.log(self.gain()/2).square()


class MixedLinear(nn.Module):
    def __init__(self,base,factors,coeff,layer,scale):
        super().__init__();self.base=base;self.coeff=coeff;self.layer=layer;self.scale=scale
        self.register_buffer('a',torch.stack([x[0] for x in factors]))
        self.register_buffer('b',torch.stack([x[1] for x in factors]))
        for p in self.base.parameters():p.requires_grad_(False)
    def forward(self,x):
        weights=self.coeff.weights()[self.layer]
        y=self.base(x)
        for i in range(4):
            value=torch.nn.functional.linear(torch.nn.functional.linear(x,self.a[i]),self.b[i])
            y=y+value*(weights[i]*self.coeff.gain()*self.scale)
        return y
