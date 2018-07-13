import torch
from torch import nn


class AttentionModule(nn.Module):
    def __init__(self,in_channels,size1,size2,size3):
        super(AttentionModule,self).__init__()

        self.top= nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=0),
            nn.ReLU()
        )

        self.down = nn.Sequential(
            nn.UpsamplingBilinear2d(size=size3),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=size2),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size1),
            nn.Sigmoid()
        )



    def forward(self,x):
        weight = self.top(x)
        weight = self.down(x)
        print(weight.shape)
        x = weight * x
        print(x.shape)
        return x

if __name__ =='__main__':
    a = torch.randn( (1,3,38,38))
    attend = AttentionModule(3,(38,38),(19,19),(10,10))
    attend(a)
