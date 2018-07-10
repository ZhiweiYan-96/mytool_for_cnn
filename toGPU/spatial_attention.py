import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    def __init__(self,in_channel=5):
        super(SpatialAttentionLayer,self).__init__()
        self.conv = nn.Conv2d(in_channel,1,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        weight_map = F.sigmoid(self.conv(x))
        #print('x:'+str(x))
        #print('wieght map:'+str(weight_map))
        result = weight_map * x
        #print('result:'+str(result))
        return weight_map * x


#test code
if __name__ == '__main__':
    sp_at = SpatialAttention(in_channel=5)
    a = torch.randn(2,5,3,3)
    sp_at(a)
