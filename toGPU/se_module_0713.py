import torch
from torch import nn


class AttentionModule(nn.Module):
    def __init__(self,in_channels,size1,size2,size3):
        super(AttentionModule,self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        print('size1:'+str(size1))
        self.top= nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels,in_channels//2,kernel_size=3,stride=1,padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
        )

        self.down = nn.Sequential(
            nn.UpsamplingBilinear2d(size=size2),
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=size1),
            nn.Conv2d(in_channels//2,in_channels,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()
        )

        self.top1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels,in_channels//2,kernel_size=3,stride=1,padding=0),
            nn.Sigmoid(),
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=0),
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )



    def forward(self,x):
        if self.size1==(10,10):
            weight = self.top1(x)
        else:
            weight = self.top(x)
        weight = self.down(weight)
        print('weight shape:'+str(weight.shape))
        x = weight * x
        print(x.shape)
        return x

if __name__ =='__main__':
    a = torch.randn( (1,3,10,10))
    attend = AttentionModule(3,(10,10),(5,5),(3,3))
    attend(a)
