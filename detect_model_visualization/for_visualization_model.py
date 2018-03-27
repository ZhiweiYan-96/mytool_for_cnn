from __future__ import print_function
import sys
import cv2
import numpy
import argparse
import lmdb
sys.path.append('/home/yanzhiwei/caffe/python/')
import caffe
from matplotlib import pyplot

class LMDB:
    def __init__(self,lmdb_path):
	'''
		Constructor, given lmdb path.

		:param lmdb_path:path to lmdb
		: type lmdb_path: string
	'''
	self._lmdb_path=lmdb_path
	self._write_pointer=0
	'''(int) Pointer for writing and appending'''

    def read(self,key = ''):
	'''
	Reading s single element or the wholse LMDB depending on wehter key is specified.Essentially a prox for
	:func 'lmdb.LMDB.read_single'
	and : func 'lmdb.LMDB.read_all'

	:param key: key s 8-digit string of the entyr to read
	:type key : string
	:return :data and labels from the LMDB as associate dictionaries,where  the key as string is the dictornary key and the value the numpy.ndarray for the data and the label for the labels
	:rtype ( {string:numpy.ndarray},{string:float} )
	'''

        if not key:
            return self.read_all()
        else:
            return self.read_single(key)

    def read_single(self,key):
	image=False
	label=False
	env=lmdb.open(self._lmdb_path,readonly=True)

	with evn.begin() as transaction:
		raw=transaction.get(key)
		datum=caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(raw)

		label=datum.label
		if datum.data:
			image=numpy.fromstring(datum.data,dtype=numpy.uint8).reshape(datum.channels,datum.height,datum.width).transpose(1,2,0)
	return image,label,key

    def read_all(self):
        '''
            read thw wholse lmdb. the method will return the data and alables (if applicable )
            as dictornary wihci is indexed by the eight-dight numbers sotred as strings

            :return: images,labels,and corresponding keys
            : rtype: ([numpy.ndarray],[int],[string]  )
        '''
        images=[]
        labels=[]
        keys=[]
        env=lmdb.open(self._lmdb_path,readonly=True)

        with env.begin() as transaction:
            cursor=transaction.cursor()

            for key, raw in cursor:
                datum=caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw)

                label=datum.label

                if datum.data:
                    image=numpy.fromstring(datum.data,dtype=numpy.uint8).reshape(datum.chanels,datum.height,datum.width).transpose(1,2,0)
                else :
                    image=numpy.array(datum.float_data).astype(numpy.float).reshape(datum.channels,datum.height,datum.width).transpose(1,2,0)

                images.append(image)
                labels.append(label)
                keys.append(key)

            return images,labels,keys

    def count(self):
        '''
            Get the number of elemebts in the lmdb

            :return:count of elements
            :rtype: int
        '''
        env=lmdb.open(self._lmdb_path)
        with env.begin() as transaction:
                return transaction.stat()['entries']

    def keys(self,n=0):
        '''
        Get the first n(or all) keys of the LMDB

        :param n: number of keys to get ,0 to get all keys
        :type n : int
        :return : list of keys
        :rtype: [string ]
        '''
        keys=[]
        env=lmdb.open(self._lmdb_path,readonly=True)

        with env.begin() as transaction:
            cursor=transaction.cursor()

            i=0
            for key,value in cursor:

                if i>=n and n>0 :
                    break;
                keys.append(key)
                i+=1
    def read_annotation(self):
        env=lmdb.open(self._lmdb_path,readonly=True)
	keys=[]
	datums=[]
	annotations=[]
        with env.begin() as transaction:
            cursor=transaction.cursor()
            for key,raw in cursor:
                annotateddatum=caffe.proto.caffe_pb2.AnnotatedDatum()
                annotateddatum.ParseFromString(raw)

                datum=annotateddatum.datum
                annotationgroup=annotateddatum.annotation_group
		
		keys.append(key)
		datums.append(datum)
		annotations.append(annotationgroup)
		'''
                if(datum):
                    #print('datum is exist')
		    #print(dir(datum))
                if(annotationgroup):
                    #print('annotationgrou exists')
		    #print(len(annotationgroup))
		'''


        return keys,datums,annotations



lmdb_path='/home/yanzhiwei/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb'

db=LMDB(lmdb_path)
#print db.keys(n=5)
#[imges,labels,keys]=db.read_all()
(keys,datums,annotations)=db.read_annotation()
db.read_annotation()
print ("len(keys):{}".format(len(keys)))
#print ("labels[1:50]"+str(labels[1:50]))
print ("len(datums):{}".format(len(datums)))
print ("len(annotaions)"+str(len(annotations)))
print ( dir(annotations[1][0]) )
'''
print labels[1:5]
print keys[1:5]
'''
# todo
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
label_map_file = "/home/yanzhiwei/caffe/data/VOC0712/labelmap_voc.prototxt"
weight_file="/home/yanzhiwei/caffe/models/SSD_AUTHOR_MODEL/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"
batch_size=1
test_prototxt='test.prototxt'
solver_prototxt='solver.prototxt'
'''
test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
'''

#net=caffe.Net(test_prototxt,weight_file,caffe.TEST)
#solver=caffe.get_solver(solver_prototxt)
#solver.test_net[0].forward()

'''
net.data,net.label=CreateAnnotatedDataLayer(lmdb_path,batch_size=batch_size,train=False,output_label=True,label_map_file=label_map_file,transform_param=test_transform_param)
VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

conf_name = "mbox_conf"
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
'''
