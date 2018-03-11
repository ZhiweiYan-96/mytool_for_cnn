import os 
import cPickle


import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.cross_validation

import lmdb
import sys
sys.path.append('/home/yanzhiwei/caffe/python')
import caffe

def unpickle(file):
	fo=open(file,'rb')
	dict=cPickle.load(fo)
	fo.close()
	return dict

def shuffle_data(data,labels):
	data,_,labels=sklearn.cross_validation.tran_test_split(
		data,labels,test_size=0.0,random_state=42
	)
	return data,labels

	
def load_data(train_file):
	d=unpickle(train_file)
	data=d['data']
	fine_labels=d['fine_labels']
	length=len(d['fine_labels'])
	
	#data,labels=shuffle_data(data,np.array(fine_labels))
	return ( data.reshape(length,3,32,32), fine_labels )
	
def change_to_lmdb(input_file, output_file):
	
	data,labels=load_data(input_file)
	
	print('Data is fully loaded, now truly convertin')
	
	env=lmdb.open(output_file,map_size=50000*1000*5)
	txn=env.begin( write=True)
	count=0
	for i in range( data.shape[0] ):
		datum=caffe.io.array_to_datum(data[i],labels[i])
		str_id='{:08}'.format(count)
		txn.put(str_id,datum.SerializeToString())
		
		count+=1
		#print(count)
		if count%1000==0:
			print('already handled with {} pictrues'.format(count))
			txn.commit()
			txn=env.begin(write=True)
	txn.commit()
	env.close()
	
	
if __name__=='__main__':
	
	train_python='/home/yanzhiwei/data/cifar-100-python/train'
	test_python='/home/yanzhiwei/data/cifar-100-python/test'
	
	train_output='/home/yanzhiwei/data/cifar_lmdb/train'
	test_output='/home/yanzhiwei/data/cifar_lmdb/test'
	
	change_to_lmdb(train_python,train_output)
	change_to_lmdb(test_python,test_output)
