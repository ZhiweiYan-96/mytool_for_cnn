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

def get_information(content):
	iterations=[]
	loss=[]
	mbox_loss=[]
	detection_eval=[]
	p=re.compile(r'Iteration ([0-9]+), loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
			#print m.group(1)
			#print m.group(2)
		iterations.append(m.group(1))
		loss.append(m.group(2))

	p=re.compile(r'mbox_loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
		mbox_loss.append(m.group(1))

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

	return (iterations,loss,mbox_loss,detection_eval)

if __name__=="__main__":
	file_name1=sys.argv[1]
	file_name2=sys.argv[2]
	#pattern=re.compiler('Iteration \d000,loss = [\d+]')
	#strlist=re.findall(r'Iteration [0-9]+, loss = [[0-9]+\.[0-9]+]',content)
	#print strlist
	#for i in range(0,len(strlist)):

	#print iterations
	#print loss

	#print mbox_loss

	content1=readContent(file_name1)
	content2=readContent(file_name2)

	(iterations1,loss1,m_box_loss1,detection_eval1)=get_information(content1)
	(iterations2,loss2,m_box_loss2,detection_eval2)=get_information(content2)
	'''
	print len(iterations)
	print len(loss)
	print len(mbox_loss)
	'''
	print len(iterations1)
	#plt.subplot(121)
	plt.plot(iterations1,loss1,'b',label=file_name1+" loss")
	plt.xlabel('iterations',fontsize=15)
	plt.ylabel('loss',fontsize=15)
	plt.plot(iterations2,loss2,'r',label=file_name2+" loss")
	plt.title('iteration vs loss')
	plt.legend()
	plt.show()


	'''
	loss_min_ind=np.argmin(loss)
	print loss.shape
	print loss[loss_min_ind]
	print 'loss_min_ind:'
	print loss_min_ind
	m_box__loss_min_ind=np.argmin(m_box__loss)
	if (len(detection_eval)>0):
		detection_eval_max_ind=np.argmax(detection_eval)
	'''



#plt.annotate(s='minimum loss={}'.format(np.min(loss)),xy=(iterations[loss_min_ind]-200,loss[loss_min_ind]+5]),xytext=(iterations[loss_min_ind],loss[loss_min_ind]),arrowprops=dict(facecolor='#AA0000',shrink=0.05),horizontalalignment='right',verticalalignment='top')
	#plt.annotate(s='minimum loss={},iterations={}'.format(np.min(loss),iterations[loss_min_ind]),xytext=(iterations[loss_min_ind]-200,loss[loss_min_ind]+5),xy=( iterations[loss_min_ind], loss[loss_min_ind] ) , arrowprops=dict( facecolor='red', arrowstyle="->" ), horizontalalignment='right', verticalalignment='top' )
	'''
	plt.legend()
	plt.subplot(132)
	plt.plot(iterations,mbox_loss,'r',label='mbox_loss')
	plt.xlabel('iterations',fontsize=15)
	plt.ylabel('mbox_loss',fontsize=15)
	plt.title('iteration vs m_box__loss')
	plt.legend()
	'''
	'''
	if (len(detection_eval)>0):
		plt.subplot(122)
		plt.annotate(s='max mAP={},{}th test'.format(np.max(detection_eval),detection_eval_max_ind),xy=( test_index[detection_eval_max_ind],detection_eval[detection_eval_max_ind]),xytext=( test_index[detection_eval_max_ind]-3,detection_eval[detection_eval_max_ind]-5),arrowprops=dict( facecolor="black",arrowstyle="->" ) )
		plt.plot(test_index,detection_eval,'b',label='detection_eval(mAP)')
		plt.xlabel('nth test',fontsize=15)
		plt.ylabel('detection_eval(mAP)',fontsize=15)
		plt.title(' test vs detection_eval')
		plt.legend()
	plt.show()
	'''
