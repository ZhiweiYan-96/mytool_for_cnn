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
	

	
if __name__=="__main__":
	file_name=sys.argv[1]
	content=readContent(file_name)
	iterations=[]
	loss=[]
	mbox_loss=[]
	detection_eval=[]
	#pattern=re.compiler('Iteration \d000,loss = [\d+]')
	#strlist=re.findall(r'Iteration [0-9]+, loss = [[0-9]+\.[0-9]+]',content)
	#print strlist
	#for i in range(0,len(strlist)):
	p=re.compile(r'Iteration ([0-9]+), loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
		#print m.group(1)
		#print m.group(2)
		iterations.append(m.group(1))
		loss.append(m.group(2))
	#print iterations
	#print loss

	p=re.compile(r'mbox_loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
		mbox_loss.append(m.group(1))
	#print mbox_loss

	p=re.compile(r'detection_eval = ([0-9]*[.]*[0-9]*[e-]*[0-9]*)')
	for m in p.finditer(content):
		print m.group(1)
		detection_eval.append(float(m.group(1))*100)
	test_index=[ x+1 for x in range(0,len(detection_eval))]
	#detection_eval*=100
	
	iterations=np.array(iterations).astype(np.float)
	loss=np.array(loss).astype(np.float)
	m_box__loss=np.array(loss).astype(np.float)
	detection_eval=np.array(detection_eval).astype(np.float)
	
	loss_min_ind=np.argmin(loss)
	print loss.shape
	print loss[loss_min_ind]
	print 'loss_min_ind:'
	print loss_min_ind
	m_box__loss_min_ind=np.argmin(m_box__loss)
	detection_eval_max_ind=np.argmax(detection_eval)
	

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
	'''
	plt.legend()
	plt.subplot(132)
	plt.plot(iterations,mbox_loss,'r',label='mbox_loss')
	plt.xlabel('iterations',fontsize=15)
	plt.ylabel('mbox_loss',fontsize=15)
	plt.title('iteration vs m_box__loss')
	plt.legend()
	'''
	plt.subplot(133)
	plt.annotate(s='max mAP={},{}th test'.format(np.max(detection_eval),detection_eval_max_ind),xy=( test_index[detection_eval_max_ind],detection_eval[detection_eval_max_ind]),xytext=( test_index[detection_eval_max_ind]-3,detection_eval[detection_eval_max_ind]-5),arrowprops=dict( facecolor="black",arrowstyle="->" ) )
	plt.plot(test_index,detection_eval,'b',label='detection_eval(mAP)')
	plt.xlabel('nth test',fontsize=15)
	plt.ylabel('detection_eval(mAP)',fontsize=15)
	plt.title(' test vs detection_eval')
	plt.legend()
	plt.show()

	