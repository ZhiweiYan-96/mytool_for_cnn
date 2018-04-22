import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
import torch.nn.init as init
import numpy as np


class GeneralizePoolingFunction(Function):
    def __init__(self,kernel_size):
        self.kernel_size=kernel_size


    def forward(self,a,input):
        #self.save_for_backward(input,a
        #print input

        #print kernel_size
        maxp,index=nn.functional.max_pool2d( Variable(input),kernel_size=self.kernel_size,return_indices=True)
        avgp=nn.functional.avg_pool2d(Variable(input),kernel_size=self.kernel_size)
        #print 'maxp'+str(maxp)
        maxp=maxp.data
        avgp=avgp.data
        index=index.data

        #self.kernel_size=kernel_size
        self.index=index
        self.max=maxp
        self.avg=avgp


        output=a*maxp+(1-a)*avgp
        self.save_for_backward(input,a)
        return output


    def backward(self,grad_output):
        (input,a)=self.saved_tensors

        index=self.index
        max_tensor=self.max
        avg_tensor=self.avg
        kernel_size=self.kernel_size

        '''
            Implementation of gradient w.r.t a
        '''
        #2x2
        grad_a=(grad_output*(max_tensor-avg_tensor)).sum()/(kernel_size*kernel_size)
        #print 'grad_a:'+str(grad_a)
        grad_a=torch.FloatTensor([grad_a])


        '''
            Implementation of gradient w.r.t input
        '''
        number=input.shape[0]
        channel=input.shape[1]
        height=input.shape[2]
        width=input.shape[3]
        output_number=index.shape[0]
        output_channel=index.shape[1]
        output_height=index.shape[2]
        output_width=index.shape[3]

        one_function= torch.zeros( input.shape ).type(torch.FloatTensor)
        one_function=one_function.view( (channel*number*width*height) )
        index_reshape=index.view( (output_number*output_channel*output_width*output_height) )
        one_function[ index_reshape ]=1
        one_function=one_function.view((number,channel,width,height)  )


        #grad_input=a*one_function
        grad_input=a*one_function+(1-a)*1.0/(1.0*kernel_size*kernel_size)

        for n in range(0,number):
            for i in range(0,channel):
                for j in range(0,width):
                    for k in range(0,height):
                        grad_input[n,i,j,k]=(grad_output[n,i,(j/kernel_size),(k/kernel_size)])*grad_input[n,i,j,k]

        #print 'out'


        #print grad_a
        #print grad_input
        return grad_a,grad_input


class Generalize_Pool2d(nn.Module):
    def __init__(self,kernel_size):
        super(Generalize_Pool2d,self).__init__()
        self.kernel_size=kernel_size
    def forward(self,a,input):
        return GeneralizePoolingFunction(self.kernel_size)(a,input)

'''
if __name__=='__main__':
    x=Variable( torch.randn(1,4, 4,).type(torch.DoubleTensor), requires_grad=True )
    a= Variable( torch.DoubleTensor([.5]) )
    #kernel_size=Variable( torch.IntTensor([2]) )
    kernel_size=2
    #gen_pool=GeneralizePoolingFunction(kernel_size=2)
    gen_pool=Generalize_Pool2d(kernel_size)
    out_grad=Variable( torch.randn(1,2,2).type(torch.DoubleTensor) )
    y=gen_pool(a,x)
    print y.backward(out_grad)

#    print gradcheck(GeneralizePoolingFunction.apply,x)


    #print test
'''
