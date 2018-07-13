import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(512, 20)
        self.L2Norm=L2Norm(512,20)
        self.extras = nn.ModuleList(extras)

        self.up_fc7 = nn.ConvTranspose2d( 1024,512, kernel_size=2,stride=2,padding=0)
        self.up_fc7_conv = nn.Conv2d(512,512,stride=1,kernel_size=3,padding=1)
        #self.down_conv3_1 = nn.Conv2d(256,512,stride=1,kernel_size=1)
        self.down_conv3_1 = nn.Conv2d(256,512,stride=1,kernel_size=1)
        self.down_conv3_1_conv = nn.Conv2d(512,512,stride=1,kernel_size=3,padding=1)
        self.batch_norm0 = nn.BatchNorm2d(512)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.se0 = AttentionModule(512*3,(38,38),(19,19),(10,10))
        self.scale0 = ScaleLayer(init_value=1)

        self.down_conv4_1 = nn.Conv2d(512,1024,stride=1,kernel_size=1)
        self.down_conv4_1_conv = nn.Conv2d(1024,1024,stride=1,kernel_size=3,padding=1)
        self.up_conv6_2 = nn.ConvTranspose2d(512,1024,kernel_size=3,stride=2,padding=1)
        self.up_conv6_2_conv = nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)
        self.batch_norm3 = nn.BatchNorm2d(1024)
        self.batch_norm4 = nn.BatchNorm2d(1024)
        self.batch_norm5 = nn.BatchNorm2d(1024)
        self.se1 = AttentionModule(1024*3,(19,19),(10,10),(5,5))
        self.scale1 = ScaleLayer(init_value=1)

        self.down_fc7 = nn.Conv2d(1024,512,stride=1,kernel_size=1,padding=0)
        self.down_fc7_conv = nn.Conv2d(512,512,stride=1,kernel_size=3,padding=1)
        self.up_conv7_2 = nn.ConvTranspose2d(256,512,stride=2,kernel_size=2,padding=0)
        self.up_conv7_2_conv = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.batch_norm7 = nn.BatchNorm2d(512)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.se2 = AttentionModule(512*3,(10,10),(5,5),(3,3))
        self.scale2 = ScaleLayer(init_value=1)


        self.down_conv6_2 = nn.Conv2d(512,256,stride=1,kernel_size=1,padding=0)
        self.down_conv6_2_conv = nn.Conv2d(256,256,stride=1,kernel_size=3,padding=1)
        self.up_conv8_2 = nn.ConvTranspose2d(256,256,stride=2,padding=1,kernel_size=3)
        self.up_conv8_2_conv = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.batch_norm9 = nn.BatchNorm2d(256)
        self.batch_norm10 = nn.BatchNorm2d(256)
        self.batch_norm11 = nn.BatchNorm2d(256)
        self.se3 = AttentionModule(256*3,(5,5),(3,3),(1,1))
        self.scale3 = ScaleLayer(init_value=1)

        self.down_conv7_2_conv = nn.Conv2d(256,256,stride=1,kernel_size=3,padding=1)
        self.up_conv9_2 = nn.ConvTranspose2d(256,256,stride=1,kernel_size=3,padding=0)
        self.up_conv9_2_conv = nn.Conv2d(256,256,stride=1,padding=1,kernel_size=3)
        self.batch_norm12 = nn.BatchNorm2d(256)
        self.batch_norm13 = nn.BatchNorm2d(256)
        self.batch_norm14 = nn.BatchNorm2d(256)
        #self.se4 = AttentionModule(256*3)
        self.scale4 = ScaleLayer(init_value=1)



        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
            if k == 11:
                conv3_1 = x
            if k == 18:
                conv4_1 = x
        conv4_3 = x

        '''
        s = self.L2Norm(x)
        sources.append(s)
        '''
        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)


        #merge information from conv3_1, conv4_3,fc7
        #reduce dimension of fc7,then apply deconv on fp,then conv3x3
        fc7 = x
        fc7_up =F.relu(self.up_fc7( fc7 ))
        fc7_up =F.relu(self.up_fc7_conv( fc7_up ))
        conv3_1_down = F.max_pool2d(conv3_1,kernel_size=2,stride=2,ceil_mode=True)
        conv3_1_down =F.relu(self.down_conv3_1( conv3_1_down ))
        conv3_1_down = F.relu(self.down_conv3_1_conv( conv3_1_down))
        conv3_1_down = self.batch_norm0(conv3_1_down)
        conv4_3_bn = self.batch_norm1( conv4_3 )
        fc7_up = self.batch_norm2( fc7_up )
        conv4_3_bn = torch.cat([conv4_3_bn,fc7_up,conv3_1_down],dim=1)
        conv4_3_bn = self.se0( conv4_3_bn )
        [conv4_3_bn , fc7_up , conv3_1_down] = torch.split(conv4_3_bn, split_size_or_sections=512,dim=1)
        conv4_3_bn= conv4_3_bn + fc7_up + conv3_1_down
        conv4_3_1 = self.L2Norm( conv4_3_bn )
        sources.append(conv4_3_1)




        '''
        sources.append(x)
        Append fc7, however need merge
        '''
        pred_last_4 = []
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                pred_last_4.append(x)
            if k == 1:
                conv6_2 = x
            if k == 3:
                conv7_2 = x
            if k == 5:
                conv8_2 = x
            if k == 7:
                conv9_2 = x

        conv4_1_down = F.max_pool2d(conv4_1,kernel_size=2,stride=2,ceil_mode=True)
        conv4_1_down = F.relu( self.down_conv4_1( conv4_1_down ) )
        conv4_1_down = F.relu( self.down_conv4_1_conv(conv4_1_down) )
        conv4_1_down = self.batch_norm3( conv4_1_down)
        fc7_bn = self.batch_norm4( fc7 )
        conv6_2_up = self.up_conv6_2( conv6_2 )
        conv6_2_up = self.up_conv6_2_conv(conv6_2_up)
        conv6_2_up  = self.batch_norm5(conv6_2_up)
        fc7_1 = torch.cat([fc7_bn,conv4_1_down,conv6_2_up],dim=1)
        fc7_1 = self.se1( fc7_1)
        [fc7_bn, conv4_1_down, conv6_2_up] = torch.split(fc7_1,split_size_or_sections=1024,dim=1)
        fc7_1 = fc7_bn + conv4_1_down + conv6_2_up
        fc7_1 = self.scale1( fc7_1 )
        sources.append( fc7_1 )

        #merge conv6_2
        fc7_down = F.max_pool2d(fc7,kernel_size=2,stride=2,ceil_mode=True)
        fc7_down = F.relu( self.down_fc7( fc7_down ))
        fc7_down = F.relu( self.down_fc7_conv( fc7_down ))
        fc7_down = self.batch_norm6( fc7_down )
        conv6_2_bn = self.batch_norm7( conv6_2 )
        conv7_2_up = self.up_conv7_2( conv7_2 )
        conv7_2_up = self.up_conv7_2_conv( conv7_2_up)
        conv7_2_up = self.batch_norm8(conv7_2_up)
        conv6_2_1 = torch.cat( [conv6_2_bn, conv7_2_up,fc7_down],dim=1 )
        conv6_2_1 = self.se2(conv6_2_1)
        [conv6_2_bn, fc7_down, conv7_2_up ] = torch.split( conv6_2_1,split_size_or_sections=512,dim=1)
        conv6_2_1 = conv6_2_bn + fc7_down + conv7_2_up
        conv6_2_1 = self.scale2(conv6_2_1)
        sources.append(conv6_2_1)

        #merge conv7_2
        conv6_2_down = F.max_pool2d(conv6_2, kernel_size=2,stride=2,ceil_mode=True)
        conv6_2_down = F.relu( self.down_conv6_2(conv6_2_down) )
        conv6_2_down = F.relu( self.down_conv6_2_conv( conv6_2_down))
        conv6_2_down = self.batch_norm9( conv6_2_down )
        conv7_2_bn = self.batch_norm10( conv7_2)
        conv8_2_up = self.up_conv8_2( conv8_2 )
        conv8_2_up = self.up_conv8_2_conv( conv8_2_up )
        conv8_2_up = self.batch_norm11(conv8_2_up)
        conv7_2_1 =torch.cat( [conv7_2_bn , conv8_2_up, conv6_2_down],dim=1)
        conv7_2_1 = self.se3( conv7_2_1 )
        [conv7_2_bn, conv8_2_Up, conv6_2_down ] = torch.split(conv7_2_1,split_size_or_sections=256,dim=1)
        conv7_2_1 = conv7_2_bn + conv8_2_up + conv6_2_down
        conv7_2_1 = self.scale3( conv7_2_1 )
        sources.append( conv7_2_1 )

        #merge conv8_2
        conv7_2_down = F.max_pool2d( conv7_2, kernel_size=2,stride=2,ceil_mode=True)
        conv7_2_down = F.relu( self.down_conv7_2_conv( conv7_2_down))
        conv7_2_down = self.batch_norm12( conv7_2_down )
        conv8_2_bn = self.batch_norm13(conv8_2)
        conv9_2_up = self.up_conv9_2( conv9_2 )
        conv9_2_up = self.up_conv9_2_conv( conv9_2_up )
        conv9_2_up = self.batch_norm14(conv9_2_up)
        '''
        conv8_2_1 = torch.cat([conv7_2_down,conv8_2_bn,conv9_2_up],dim=1)
        conv8_2_1 = self.se4( conv8_2_1 )
        [conv8_2_bn , conv9_2_up, conv7_2_down] = torch.split(conv8_2_1,split_size_or_sections=256,dim=1)
        '''
        conv8_2_1 = conv8_2_bn + conv9_2_up + conv7_2_down
        conv8_2_1 = self.scale4( conv8_2_1 )
        sources.append( conv8_2_1 )

        sources.append(conv9_2)

        #merge conv4_3 downsample conv3_1, upsample fc7,

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
