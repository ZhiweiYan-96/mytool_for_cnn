import sys
import numpy as np
import matplotlib.pyplot as plt
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
		#print m.group(1)
		detection_eval.append(float(m.group(1))*100)
	test_index=[ x+1 for x in range(0,len(detection_eval))]
	#detection_eval*=100

	iterations=np.array(iterations).astype(np.float)
	loss=np.array(loss).astype(np.float)
	m_box__loss=np.array(loss).astype(np.float)
	detection_eval=np.array(detection_eval).astype(np.float)

	return (iterations,loss,mbox_loss,detection_eval)

def find_index(itev_1,iteration):
    for i in np.array(range(0,len(iteration)))*-1:
        if iteration[i] == itev_1:
            return i

if __name__ == "__main__":
    file_names=[]
    for name in sys.argv:
        if name!= sys.argv[0]:
            file_names.append(name)

    iterations=[]
    losses=[]
    mbox_losses=[]
    detection_evals=[]

    for name in file_names:
        content=readContent(name)
        (iteration,loss,mbox_loss,detection_eval)=get_information(content)
        iterations.append(iteration)
        losses.append(loss)
        mbox_losses.append(mbox_loss)
        detection_evals.append(detection_eval)
    indexes=[]
    for i in range(0,len(file_names)-1):
        index=find_index( iterations[i+1][0],iterations[i] )
        iterations[i]=(iterations[i])[0:index-1]
        losses[i]=(losses[i])[0:index-1]
        mbox_losses[i]=(mbox_losses[i])[0:index-1]

    iterations_total=[]
    loss_total=[]
    mbox_loss_total=[]

    iterations_total=iterations[0]
    loss_total=losses[0]
    mbox_loss_total=mbox_losses[0]
    for i in range(0,len(file_names)-1):
        iterations_total=np.hstack( ( iterations_total ,iterations[i+1]) )
        loss_total=np.hstack( (loss_total,losses[i+1]) )
        mbox_loss_total=np.hstack( (mbox_loss_total,mbox_losses[i+1]) )

    plt.plot( iterations_total,loss_total )
    #plt.show()


    f=open('total_log.txt','w')
    for i in range( 0, len(iterations_total) ):
        f.write( 'iterations:'+iterations_total[i].astype(np.str)+' '+ 'loss:'+ loss_total[i].astype(np.str)+'\n' )


    '''
    (iteration,loss,mbox_loss,detection_eval)=get_information(content)
    (iteration1,loss1,mbox_loss1,detection_eval1)=get_information(content1)
    print iteration[-1],iteration[-2]
    print iteration1[0]
    index=find_index(iteration1[0],iteration)
    print index
    print iteration[index]

    #print len(iteration[0:index])
    iteration_total=iteration[0:index-1]
    loss_total=loss[0:index-1]
    iteration_total=np.hstack((iteration_total,iteration1))
    loss_total=np.hstack((loss_total,loss1))
    print iteration_total.shape
    print loss_total.shape
    plt.plot(iteration_total,loss_total,label='totalloss')
    plt.show()
    '''
