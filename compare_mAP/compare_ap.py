import numpy
import matplotlib.pyplot as plt 
import argparse


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

parser=argparse.ArgumentParser(description="Script that compare aps of two model")
parser.add_argument('--file1',type=str,help="file 1")
parser.add_argument('--file2',type=str,help="file 2")
args=parser.parse_args()
if __name__=="__main__":

	file1=args.file1
	file2=args.file2
	(file1_names,file1_aps,file1_mAP)=parse_file(file1)
	(file2_names,file2_aps,file2_mAP)=parse_file(file2)
	plt.plot(file1_names,file1_aps,label=file1+'_'+str(file1_mAP))
	plt.plot(file2_names,file2_aps,label=file2+'_'+str(file2_mAP))
	plt.legend()
	plt.show()

	'''
	(wide_name,wide_aps,wide_mAP)=parse_file(i)
	(baseline_name,baseline_aps,baseline_mAP)=parse_file('baseline_ap.txt')
	(deep_name,deep_aps,deep_mAP)=parse_file('deep_ap.txt')
	#plt.plot(wide_name,wide_aps,label='wide_'+str(wide_mAP))
	plt.plot(deep_name,deep_aps,label='deep_'+str(deep_mAP))
	plt.plot(baseline_name,baseline_aps,label='baseline_'+str(baseline_mAP))
	
	plt.legend()
	plt.show()
	'''