import torch
import torch.nn as nn

class ScaleLayer(nn.Module):
    def __init__(self,init_value=1e-3):
        super(ScaleLayer,self).__init__()
        self.scale = nn.Parameter( torch.FloatTensor([init_value]))

    def forward(self,input):
        return input * self.scale
