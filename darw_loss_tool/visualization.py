import sys
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np

import re

def readContent(file_name='ssd_up.log'):
	f=file(file_name,'r')
	content=f.read()
	f.close()
	return content

def read_loss_eval(content):
	iterations=[]
	loss=[]
	detection_eval=[]
	p=re.compile(r'Iteration ([0-9]+), loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
		iterations.append(m.group(1))
		loss.append(m.group(2))
	
	p=re.compile(r'detection_eval = ([0-9]*[.]*[0-9]*[e-]*[0-9]*)')
	for m in p.finditer(content):
		detection_eval.append(m.group(1))
	test_index=[ x+1 for x in range(0,len(detection_eval))]
	
	iterations=np.array(iterations).astype(np.float)
	loss=np.array(loss).astype(np.float)
	detection_eval=np.array(detection_eval).astype(np.float)
	detection_eval=detection_eval*100
	test_index=np.array(test_index).astype(np.float)
	return (iterations,loss,detection_eval,test_index)


	
	
if __name__=="__main__":
	'''
	iterations=[]
	loss=[]
	detection_eval=[]
	test_index=[]
	'''
	file_name=sys.argv[1]
	content=readContent(file_name)
	(iterations,loss,detection_eval,test_index)=read_loss_eval(content)
	
	'''
	iterat1=[]
	loss1=[]
	detection_eval1=[]
	test_index1=[]
	'''
	file_name1=sys.argv[2]
	content1=readContent(file_name1)
	(iterations1,loss1,detection_eval1,test_index1)=read_loss_eval(content1)
	plt.subplot(131)
	plt.plot(iterations,loss,'b',label='{} loss'.format(file_name))
	plt.plot(iterations1,loss1,'r',label='{} loss'.format( file_name1 ))
	plt.legend()
	
	
	plt.subplot(132)
	plt.plot(test_index,detection_eval,'b',label='{} mAP'.format(file_name))
	plt.legend()
	
	plt.subplot(133)
	plt.plot(test_index1,detection_eval1,'r',label='{} mAP'.format(file_name1))
	plt.legend()
	plt.show()
	
'''
	print len(iterations)
	print len(loss)
	print len(mbox_loss)
	plt.subplot(131)
	plt.plot(iterations,loss,'b',label='loss')
	plt.xlabel('iterations',fontsize=15)
	plt.ylabel('loss',fontsize=15)
	plt.title('iteration vs loss')
#plt.annotate(s='minimum loss={}'.format(np.min(loss)),xy=(iterations[loss_min_ind]-200,loss[loss_min_ind]+5]),xytext=(iterations[loss_min_ind],loss[loss_min_ind]),arrowprops=dict(facecolor='#AA0000',shrink=0.05),horizontalalignment='right',verticalalignment='top')
	plt.annotate(s='minimum loss={},iterations={}'.format(np.min(loss),iterations[loss_min_ind]),xytext=(iterations[loss_min_ind]-200,loss[loss_min_ind]+5),xy=( iterations[loss_min_ind], loss[loss_min_ind] ) , arrowprops=dict( facecolor='red', arrowstyle="->" ), horizontalalignment='right', verticalalignment='top' )
	plt.legend()
	plt.subplot(132)
	plt.plot(iterations,mbox_loss,'r',label='mbox_loss')
	plt.xlabel('iterations',fontsize=15)
	plt.ylabel('mbox_loss',fontsize=15)
	plt.title('iteration vs m_box__loss')
	plt.legend()
	plt.subplot(133)
	plt.annotate(s='max mAP={},{}th test'.format(np.max(detection_eval),detection_eval_max_ind),xy=( test_index[detection_eval_max_ind],detection_eval[detection_eval_max_ind]),xytext=( test_index[detection_eval_max_ind]-3,detection_eval[detection_eval_max_ind]-5),arrowprops=dict( facecolor="black",arrowstyle="->" ) )
	plt.plot(test_index,detection_eval,'b',label='detection_eval(mAP)')
	plt.xlabel('nth test',fontsize=15)
	plt.ylabel('detection_eval(mAP)',fontsize=15)
	plt.title(' test vs detection_eval')
	plt.legend()
	plt.show()

'''