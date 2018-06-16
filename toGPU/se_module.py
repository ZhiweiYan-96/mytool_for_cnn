import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self,channel):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d( (9,9) )
        self.conv1 = nn.Conv2d(512,512,kernel_size=3,stride=3,padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512,512,kernel_size=3,stride=3,padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
                nn.Conv2d( 512 , 512, kernel_size=3, stride=3, padding=0),
                nn.ReLU(inplace=True),
                #nn.Linear( channel, channel  // reduction ),
                nn.Conv2d( 512, 512,  kernel_size=3, stride=3, padding=0),
                nn.ReLU( inplace=True ),
                #nn.Linear( channel  // reduction ,channel)
                nn.Sigmoid()
        )


    def forward( self, x ):
        b, c , _ , _ = x. size()
        #y = self.avg_pool( x ).view( b , c)
        #y = self.fc( y ).view( b , c , 1 ,1 )
        y = self.avg_pool(x)
        print(y.shape)
        conv1 = self.conv1(y)
        conv1 = self.relu1( conv1 )
        print(conv1.shape)
        conv2 = self.relu2( self.conv2(conv1) )
        sigmoid = self.sigmoid( conv2 )
        print(sigmoid.shape)
        sigmoid = sigmoid.view(b,c,1,1)
        return sigmoid * x

if __name__ =='__main__':
    se_module = SELayer(512)
    data = torch.ones( (1,512,38,38),dtype=torch.double)
    print(data.shape)
    result=se_module(data.float())
