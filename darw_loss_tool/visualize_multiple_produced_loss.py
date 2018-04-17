import sys
import numpy as np
import matplotlib.pyplot as plt
import re

def readContent(file_name='total_log.txt'):
	f=file(file_name,'r')
	content=f.read()
	f.close()
	return content

def get_information(content):
	iterations=[]
	loss=[]
	mbox_loss=[]
	detection_eval=[]
	p=re.compile(r'iterations:([0-9]+[\.]*[0-9]*) loss:([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
			#print m.group(1)
			#print m.group(2)
		iterations.append(m.group(1))
		loss.append(m.group(2))

        '''
	p=re.compile(r'mbox_loss = ([0-9]+[\.]*[0-9]*)')
	for m in p.finditer(content):
		mbox_loss.append(m.group(1))

	p=re.compile(r'detection_eval = ([0-9]*[.]*[0-9]*[e-]*[0-9]*)')
	for m in p.finditer(content):
		#print m.group(1)
		detection_eval.append(float(m.group(1))*100)
	test_index=[ x+1 for x in range(0,len(detection_eval))]
	#detection_eval*=100
        '''

	iterations=np.array(iterations).astype(np.float)
	loss=np.array(loss).astype(np.float)
	#m_box__loss=np.array(loss).astype(np.float)
	#detection_eval=np.array(detection_eval).astype(np.float)
	return (iterations,loss)

def find_index(itev_1,iteration):
    for i in np.array(range(0,len(iteration)))*-1:
        if iteration[i] == itev_1:
            return i

if __name__=="__main__":
    filenames=[]
    for name in sys.argv:
        if name!= sys.argv[0]:
            filenames.append(name)
    iterations=[]
    losses=[]
    for name in filenames:
        content=readContent(name)
        (iteration,loss)=get_information(content)
        iterations.append(iteration)
        losses.append(loss)

    for i in range(0,len(filenames)):
        plt.plot(iterations[i],losses[i],label=filenames[i])
    plt.legend()
    plt.show()
