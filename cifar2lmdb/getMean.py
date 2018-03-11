import sys
sys.path.append('/home/yanzhiwei/caffe/python')
import lmdb
import numpy as np 
from caffe.proto import caffe_pb2
import time

lmdb_env=lmdb.open('cifar100_train_lmdb')
lmdb_txn=lmdb_env.begin()
lmdb_cursor=lmdb_txn.curosr()
datum=caffe_pb2.Datum()

N=0
mean=np.zeros( (1,3,32,32) )
beginTime=time.time()
for key,value in lmdb_curosr:
	datum.ParseFromString(value)
	data=caffe.io.datum_to_array(datum)
	image=data.transpose(1,2,0)
	mean[0,0]+=image[:,:,0]
	mean[0,1]+=image[:,:,1]
	mean[0,2]+=image[:,:,2]
	N+=1
	if N%1000 == 0 :
		elapsed=time.time()-begTime
		print("Processed {} image in {:.2f} seconds. {:.2f} image/second".format(N,elapsed,N/elapsed) )

mean[0]/=N
blob=caffe.io.array_to_blobproto(mean)
with open ('mean.binaryproto','wb') as f:
	f.write(blob.SerializeToString())

lmdb_env.close
		