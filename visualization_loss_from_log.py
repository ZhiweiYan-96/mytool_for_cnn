import numpy as np 
import matplotlib.pyplot as plt

import re

def readContent():
	f=file('log.txt','r')
	content=f.read()
	f.close()
	return content
	
content=readContent()
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
	detection_eval.append(m.group(1))
test_index=[ x+1 for x in range(0,len(detection_eval))]
	

print len(iterations)
print len(loss)
print len(mbox_loss)
plt.subplot(131)
plt.plot(iterations,loss,'b',label='loss')
plt.xlabel('iterations',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.title('iteration vs loss')
plt.legend()
plt.subplot(132)
plt.plot(iterations,mbox_loss,'r',label='mbox_loss')
plt.xlabel('iterations',fontsize=15)
plt.ylabel('mbox_loss',fontsize=15)
plt.title('iteration vs m_box__loss')
plt.legend()
plt.subplot(133)
plt.plot(test_index,detection_eval,'b',label='detection_eval')
plt.xlabel('nth test',fontsize=15)
plt.ylabel('detection_eval',fontsize=15)
plt.title(' test vs detection_eval')
plt.legend()
plt.show()

	