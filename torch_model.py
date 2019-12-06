#BUILDING THE CNN MODEL USING PYTORCH

import torch
import torch.nn as nn
import torch.nn.functional as F 

class Torch_Model(nn.Module):
   
    def __init__(self):
        super(Torch_Model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=6, stride = 6)
        self.dropout1= nn.Dropout(0.25)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,9))
        self.pool2=nn.MaxPool2d(kernel_size=2, stride =2)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(2,4))
        self.dropout2= nn.Dropout(0.5)
        self.conv5=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1)
        self.conv6=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1)


    def forward(self,x):

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool1(x)
        x=self.dropout1(x)
        x=F.relu(self.conv3(x))
        x=self.pool2(x)
        x=F.relu(self.conv4(x))
        x=self.dropout2(x)
        x=F.relu(self.conv5(x))

        x=torch.sigmoid(self.conv6(x))
    
        return x

print("torch model go")