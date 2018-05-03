import numpy
import matplotlib.pyplot as plt 


def parse_file(file_name):
	file=open(file_name)
	names=[]
	aps=[]
	for line in file.readlines():
		words=line.split()
		if words[0] != 'Mean':
			names.append(words[2])
			aps.append(float(words[4]))
			print(float(words[4]))
		else:
			meanAP=words[3]
	print(names)
	print(aps)
	return names,aps,meanAP


if __name__=="__main__":
	(wide_name,wide_aps,wide_mAP)=parse_file('wide_ap.txt')
	(baseline_name,baseline_aps,baseline_mAP)=parse_file('baseline_ap.txt')
	(deep_name,deep_aps,deep_mAP)=parse_file('deep_ap.txt')
	#plt.plot(wide_name,wide_aps,label='wide_'+str(wide_mAP))
	plt.plot(deep_name,deep_aps,label='deep_'+str(deep_mAP))
	plt.plot(baseline_name,baseline_aps,label='baseline_'+str(baseline_mAP))
	
	plt.legend()
	plt.show()