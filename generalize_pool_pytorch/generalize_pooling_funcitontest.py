import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from layers.modules.generalize_pool import Generalize_Pool2d
import torch.optim as optim




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(3,6,5)
        self.gen_pool1=Generalize_Pool2d(2,0.5)
        self.gen_pool2=Generalize_Pool2d(2,0.5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)


    def forward(self,x):

        x=self.gen_pool1(F.relu(self.conv1(x)))
        x=self.gen_pool2(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

net=Net()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

##a1=torch.FloatTensor([.5])
#a2=torch.FloatTensor([.5])
#a1=Variable(a1)
#a2=Variable(a2)

for epoch in range(2):
    running_loss=0.0

    for i, data in enumerate(trainloader,0):
        #print i
        inputs,labels=data

        inputs,labels=Variable(inputs),Variable(labels)
        optimizer.zero_grad()

        outputs=net(inputs)

        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()



        running_loss+=loss.data[0]
        if i%100 == 99:
            print('[%5d,%5d]: loss: %.3f'%(epoch+1,i+1,running_loss/100))
            print 'a1:{} a2:{}'.format(str(net.gen_pool1.a),str(net.gen_pool2.a))
            running_loss=0.0
