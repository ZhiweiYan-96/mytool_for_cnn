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

        return keys



lmdb_path='/home/yanzhiwei/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb'

db=LMDB(lmdb_path)
#print db.keys(n=5)
[imges,labels,keys]=db.read_all()
print ("len(labels):{}".format(len(labels)))
print ("labels[1:5]"+str(labels[1:5]))
print ("len(keys):{}".format(len(keys)))
print ("keys[1:5]"+str(keys[1:5]))
'''
print labels[1:5]
print keys[1:5]
'''
