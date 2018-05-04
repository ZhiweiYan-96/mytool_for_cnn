

#this mehtod compute receptive field from top to down
#formulatiion is RF=(RF-1)*stride+kernel_size
#RF=RF*stride-strde+kernel_size


#[ kernel_size,stride,padding]
net_struct = {'alexnet': {'net':[[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0]],
                   'name':['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5']},
       'vgg16': {'net':[[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],
                        [2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0]],
                 'name':['conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2',
                         'conv3_3', 'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5']},
       'zf-5':{'net': [[7,2,3],[3,2,1],[5,2,2],[3,2,1],[3,1,1],[3,1,1],[3,1,1]],
               'name': ['conv1','pool1','conv2','pool2','conv3','conv4','conv5']}}



net_struct1= {
                'ssd_300':{
                    'net':[[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],
                                     [2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0],[3,1,6],
                                     [1,1,0],[1,1,0],[3,2,1],[1,1,0],[3,2,1],[1,1,0],[3,1,0],[1,1,0],[3,1,0]],
                    'name':['conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2',
                            'conv3_3','pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','fc6',
                            'fc7','conv6_1','conv6_2','conv7_1','conv7_2','conv8_1','conv8_2','conv9_1','conv9_2']
                },
            }


imsize=300

def outFromIn(imsiz,net,layernum):
    tostride=1
    insize=imsiz
    for layer in range(layernum):
        kernel_size,stride,pad=net[layer]
        outsize=(insize-kernel_size+2*pad)/stride+1
        insize=outsize
        tostride=tostride*stride
    return outsize,tostride

def inFromOut(net,layernum):
    receptive_filed=1
    for layer in reversed( range( layernum )):
        kernel_size,stride,pad=net[ layer ]
        receptive_filed=( receptive_filed-1 )*stride+kernel_size
    return receptive_filed

if __name__=='__main__':
    print "layer output sizes given image = %dx%d" % (imsize, imsize)

    p=()
    for net in net_struct1.keys():
         print '************net structrue name is %s**************'% net
         for i in range( len( net_struct1[ net ]['net'] )):
            p=outFromIn(imsize,net_struct1[ net ][ 'net'],i+1)
            rf=inFromOut( net_struct1[net][ 'net' ],i+1)
            print "Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (net_struct1[net]['name'][i], p[0], p[1], rf)
