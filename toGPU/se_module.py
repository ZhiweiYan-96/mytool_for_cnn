import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return  x*y

if __name__ =='__main__':
    se_module = SELayer(512)
    data = torch.ones( (1,512,38,38),dtype=torch.double)
    print(data.shape)
    result=se_module(data.float())
    print(result.shape)
