import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from layers import *
from data import voc,coco,voc_321
import os
import math
import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,downsample=None, kernel=3,padding=1,dilation=1,use_bn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dilation==1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride,
                               padding=padding, bias=False)
        else:
            pad_for_dlation = int( (3+ (dilation-1)*2) -1 )/2
            self.conv2 = nn.Conv2d(planes,planes,kernel_size=kernel,stride=stride,
                                   padding=pad_for_dlation,dilation=dilation,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_bn= use_bn

    def forward(self, x):
        use_bn =self.use_bn
        residual = x

        out = self.conv1(x)

        if use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if use_bn:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




def resnet101():
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    '''
    if pretrained:
        print('loading')
        model.load_state_dict(torch.load('resnet101-5d3b4d8f.pth',map_location=lambda storage, loc: storage))
        print('Finished')
    '''
    return model

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)


        '''
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        source = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        print('layer1 size:'+str(x.shape))
        x = self.layer2(x)
        res3b3 = x
        print('layer2 size:'+str(x.shape))
        source.append(res3b3)
        x = self.layer3(x)


        self.res3b3 = res3b3




        '''
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        '''
        return x




class SSD321(nn.Module):
    def __init__(self,phase,size,base,extras,head,num_classes):
        super(SSD321,self).__init__()

        self.phase=phase
        self.num_classes= num_classes
        self.cfg = voc_321
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(),volatile=True)
        self.resnet101 = base
        self.inplanes=1024
        self.layer4 =  self._make_layer(Bottleneck,512,3,stride=1,dilation=2)
        self.extras = nn.ModuleList(extras)
        self.size=size

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])



        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes,200,0.01,0.45)

        #res_inter
        self.inters = []
        inter_downsample = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=1,stride=1),
            nn.ReLU(inplace=True)
        )
        inter_downsample1 = nn.Sequential(
            nn.Conv2d(2048,1024,kernel_size=1,stride=1),
            nn.ReLU(inplace=True)
        )
        inter_downsample2 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.inters += [ Bottleneck(inplanes=512,planes=256,stride=1,downsample=inter_downsample),
                   Bottleneck(inplanes=2048,planes=256,stride=1,downsample=inter_downsample1)]
        for k in range(0,4):
            if k==3:
                self.inters += [ Bottleneck(inplanes=1024,planes=256,stride=1,downsample=inter_downsample2,use_bn=False)]
            self.inters += [ Bottleneck(inplanes=1024,planes=256,stride=1,downsample=inter_downsample2)]

    def forward(self,x):
        source = []
        loc = []
        conf = []
        x = self.resnet101(x)
        res3b3 = self.resnet101.res3b3
        source += [ self.inters[0](res3b3) ]
        x = self.layer4(x)
        res5c = x
        source += [ self.inters[1](res5c)]
        print('res5c shape:'+str(res5c.shape))
        for k,v in enumerate(self.extras):
            print(k)
            x = v(x)
            pred = self.inters[k+2](x)
            print('extra '+str(k)+str(x.shape))
            source.append(pred)
        for (x,l,c) in zip(source,self.loc,self.conf):
            loc.append( l(x).permute(0,2,3,1).contiguous() )
            conf.append( c(x).permute(0,2,3,1).contiguous() )

        loc = torch.cat( [ o.view(o.size(0),-1) for o in loc],1)
        conf = torch.cat( [o.view(o.size(0),-1) for o in conf],1)
        print('loc shape '+str(loc.shape))
        print('conf shape'+str(conf.shape))
        if self.phase == 'test':
            output =self.detect(
                loc.view(loc.size(0),-1,4),
                self.softmax(conf.view(conf.size(0),-1,self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view( loc.size(0),-1,4 ),
                conf.view( conf.size(0),-1,self.num_classes),
                self.priors
            )
        return output


    def _make_layer(self, block, planes, blocks, stride=1,dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                     map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

mbox = {
    '321':[8,8,8,8,8,8]
}

def add_extras():
    layers = []


    res6_downsample = nn.Sequential(
        nn.Conv2d(2048,1024,kernel_size=2,stride=2,padding=0),
        nn.BatchNorm2d(1024)
    )
    res6 = Bottleneck(inplanes=2048,planes=256,stride=2,kernel=2,padding=0,downsample=res6_downsample)

    res7_downsample = nn.Sequential(
        nn.Conv2d(1024,1024,kernel_size=2,stride=2,padding=0),
        nn.BatchNorm2d(1024)
    )
    res7 = Bottleneck(inplanes=1024,planes=256,stride=2,padding=0,kernel=2,downsample=res7_downsample)

    res8_downsample = nn.Sequential(
        nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=0),
        nn.BatchNorm2d(1024)
    )
    res8 = Bottleneck(inplanes=1024,planes=256,kernel=3,stride=1,padding=0,downsample=res8_downsample)

    res9_downsample = nn.Sequential(
        nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=0),
        #nn.BatchNorm2d(1024)
    )
    res9 = Bottleneck(inplanes=1024,planes=256,kernel=3,stride=1,padding=0,downsample=res9_downsample,use_bn=False)

    layers += [res6,res7,res8,res9]

    return layers


def build_ssd(phase,size=321,num_classes=21):
    if phase != 'test' and phase != 'train':
        print('ERROR： Phase'+phase+' not recognized')
        return

    head_ = multibox(mbox,num_classes)
    base = resnet101()
    return SSD321(phase,size,base,add_extras(),head_,num_classes)

def multibox(cfg,num_classes):
    loc_layers=[]
    conf_layers=[]

    in_channels=[1024,1024,1024,1024,1024,1024]
    for i in range(0,len(in_channels)):
        if i==0:
            loc_layers += [nn.Conv2d(1024,32,stride=1,kernel_size=5,padding=2)]
            conf_layers += [nn.Conv2d(1024,21*8,stride=1,kernel_size=5,padding=2)]
        else:
            loc_layers += [nn.Conv2d(1024,32,stride=1,kernel_size=3,padding=1)]
            conf_layers += [ nn.Conv2d(1024,21*8,stride=1,kernel_size=3,padding=1)]
    return (loc_layers,conf_layers)


if __name__ == '__main__':

    net = build_ssd('train',size=320,num_classes=21)
    net.resnet101.load_state_dict(torch.load('resnet101_reduced_fc.pth'))
    x = torch.randn([1,3,320,320])
    net(x)
