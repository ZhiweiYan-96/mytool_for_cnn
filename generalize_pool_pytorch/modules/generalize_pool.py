import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from ..functions.generalize_pool2D import GeneralizePoolingFunction



class Generalize_Pool2d(Module):
    def __init__(self,kernel_size,a):
        super(Generalize_Pool2d,self).__init__()
        self.kernel_size=kernel_size
        self.a= nn.Parameter(torch.Tensor([a]))
    def forward(self,input):
        return GeneralizePoolingFunction(self.kernel_size)(self.a,input)
