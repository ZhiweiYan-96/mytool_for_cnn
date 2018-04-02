import os
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.append('/home/yan/Vision/dsod/caffe/python/')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import skimage.io as io
from xml.dom import minidom
import matplotlib.pyplot as plt

class Object:
    def __init__(self,name,xmin,ymin,xmax,ymax):
        self.name_=name
        self.xmin_=xmin
        self.ymin_=ymin
        self.xmax_=xmax
        self.ymax_=ymax

    def display(self):
        comma=', '
        #print '('+self.name_+comma+self.xmin_+comma+self.ymin_+comma+self.xmax_+comma+self.ymax_+')'

def parse_a_file(file_name):
    xmldoc=minidom.parse(file_name)
    objectlist=xmldoc.getElementsByTagName('object')
    objects=[]
    for object in objectlist:
        name=object.getElementsByTagName('name')
        bndbox=object.getElementsByTagName('bndbox')
        xmin=bndbox[0].getElementsByTagName('xmin')
        ymin=bndbox[0].getElementsByTagName('ymin')
        xmax=bndbox[0].getElementsByTagName('xmax')
        ymax=bndbox[0].getElementsByTagName('ymax')

        name_v=name[0].firstChild.data
        xmin_v=float(xmin[0].firstChild.data)
        ymin_v=float(ymin[0].firstChild.data)
        xmax_v=float(xmax[0].firstChild.data)
        ymax_v=float(ymax[0].firstChild.data)

        temp=Object(name_v,xmin_v,ymin_v,xmax_v,ymax_v)
        temp.display()
        objects.append(temp)
    return objects


def showGTBox(image_file,objects):
    img=io.imread(image_file)
    #img=img.resize()
    #plt.figure()
    #plt.subplot(121)
    img=Image.open(image_file)
    draw=ImageDraw.Draw(img)
    width,height=img.size
    for object in objects:
        name=object.name_
        xmin=object.xmin_
        ymin=object.ymin_
        xmax=object.xmax_
        ymax=object.ymax_
        draw.rectangle([xmin,ymin,xmax,ymax],outline=(0,0,255))
        draw.text([xmin,ymin],name,(0,255,0))
    img.save('gt.jpg')
    '''
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax=plt.gca()
    for object in objects:
        coords=(object.xmin_,object.ymin_),object.xmax_-object.xmin_,object.ymax_-object.ymin_
        ax.add_patch(plt.Rectangle( *coords,fill=False,linewidth=3,color='r' ))
        display_text=object.name_
        ax.text(object.xmin_,object.ymin_,display_text,color='r')
    plt.savefig("gt.jpg",bbox_inches="tight")
    plt.show()
    '''
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    img1=io.imread('result.jpg')
    #img=img.resize()
#    plt.clf()
    plt.imshow(img1)
    plt.axis('off')
    ax=plt.gca()
    res_name=os.path.splitext(image_file)[0]+"_result.png"
    plt.savefig(res_name)
    plt.show()



def get_labelname(labelmap,labels):
    num_labels=len(labelmap.item)
    labelnames=[]
    if type(labels) is not list:
        labels=[labels]
    for label in labels:
        found=False
        for i in range(0,num_labels):
            if label==labelmap.item[i].label:
                found=True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found==True
    return labelnames

class CaffeDetection:
    def __init__(self,gpu_id,model_def,model_weights,image_resize,labelmap_file):

        self.image_resize=image_resize
        self.net=caffe.Net(model_def,model_weights,caffe.TEST)
        self.transformer=caffe.io.Transformer({'data':self.net.blobs['data'].data.shape} )
        self.transformer.set_transpose('data',(2,0,1))
        self.transformer.set_mean('data',np.array([104,117,123]))
        self.transformer.set_raw_scale('data',255)
        self.transformer.set_channel_swap('data',(2,1,0))

        file=open(labelmap_file,'r')
        self.labelmap=caffe_pb2.LabelMap()
        text_format.Merge( str(file.read()) ,self.labelmap )

    def detect(self,image_file,conf_thresh=0.5,topn=5):
        self.net.blobs['data'].reshape(1,3,self.image_resize,self.image_resize)
        image=caffe.io.load_image(image_file)

        transformed_image=self.transformer.preprocess('data',image)
        self.net.blobs['data'].data[...]=transformed_image

        detections=self.net.forward()['detection_out']

        #Parse the outputs
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        #Get detctions with confidence higer than 0.6
        top_indices= [i for i, conf in enumerate(det_conf) if conf>=conf_thresh]



        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(gpu_id,model_def,model_wights,image_resize,labelmap_file):
    detection=CaffeDetection(gpu_id,model_def,model_weights,image_resize,labelmap_file)
    result=detection.detect(image_file)
    print result

    img=Image.open(image_file)
    draw=ImageDraw.Draw(img)
    width,height=img.size
    print width,height
    for item in result:
        xmin=int(round( item[0]*width ))
        ymin=int(round( item[1]*height) )
        xmax=int(round( item[2]*width) )
        ymax=int(round( item[3]*height))
        draw.rectangle( [xmin,ymin,xmax,ymax],outline=(0,255,0) )
        draw.text( [xmin,ymin],item[-1]+ str( item[-2]),(0,255,0) )
        print item
        print [xmin,ymin,xmax,ymax]
        print [xmin,ymin],item[-1]
    img.save("result.jpg")
        #k=1
#for key in range(1,9964):
#    file_name=str(key).zfill(6)
#    objects=parse_a_file(file_name+'.xml')
if __name__=='__main__':
    image_file='000001.jpg'
    model_def='deploy _ssd_up.prototxt'
    image_resize=300
    label_map_file="./labelmap_voc.prototxt"
    model_weights='VGG_VOC0712_upSSD_up300x300_iter_80000.caffemodel'
    main(1,model_def,model_weights,image_resize,label_map_file)
    objects=parse_a_file('000001.xml')
    showGTBox(image_file,objects)

    #showGTBox(image_file,objects)
