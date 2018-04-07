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
    filenames=[]
    for name in sys.argv :
        if name != sys.argv[0]:
            filenames.append(name)
	#file_name1=sys.argv[1]
	#file_name2=sys.argv[2]
	#pattern=re.compiler('Iteration \d000,loss = [\d+]')
	#strlist=re.findall(r'Iteration [0-9]+, loss = [[0-9]+\.[0-9]+]',content)
	#print strlist
	#for i in range(0,len(strlist)):

	#print iterations
	#print loss

	#print mbox_loss
    iterations=[]
    losses=[]
    mbox_losses=[]
    detection_evals=[]
    for name in filenames:
        content=readContent(name)
        (iteration,loss,mbox_loss,detection_eval)=get_information(content)
        iterations.append(iteration)
        losses.append(loss)
        mbox_losses.append(mbox_loss)
        detection_evals.append(detection_eval)

    plt.xlabel('iterations',fontsize=15)
    plt.ylabel('loss',fontsize=15)
    for i in range(0,len(iterations)):
        plt.plot( iterations[i],losses[i], label=filenames[i]+'loss'  )
        #plt.xlabel('iterations,fontsize=')
    plt.title('iteration vs loss')
    plt.legend()
    plt.show()
