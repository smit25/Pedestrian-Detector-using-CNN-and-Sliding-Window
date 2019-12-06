import torch
from torch_model import Torch_Model
import torchvision
import torch.utils.data
import torch.optim as optim 
from shvn1 import train_data, train_label, test_data, test_label,test2,test2_l

# DECLARING VARIABLES
epochs = 5
batch_size= 64
printfreq = 50

# LOADING THE MODEL
Model=Torch_Model()

#LOSS FUNCTION
lossfunc=torch.nn.BCELoss(reduction='mean')

#OPTIMIZER
optimizer=optim.Adam(Model.parameters())

# CREATING A TEXT FILE
log=open("train_write.txt","w")

#ANALYSIS
log.write("Batch Size : %d\n\n"%(batch_size))
log.write(str(Model)+'\n\n')
log.write("Optimizer : ADAM\n\n")
log.write("Epochs : %d\n\n"%(epochs))

#CUSTOM DATA GENERATOR
def custom_loader(data,label,batch):
    dataset=[]
    for i in range(len(data)):
        dataset.append([data[i],label[i]])
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=True)
    return loader


train_loader=custom_loader(train_data,train_label,batch_size)
test_loader=custom_loader(test_data,test_label,batch_size)
print("good to go loader")

# TRAINING ON THE DATASET
for e in range(0,epochs):
    
    train_loss=0.0
    running_loss=0.0
    valid_loss=0.0

    for i,data in enumerate(train_loader):
        x,y = data   
        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        output = Model(x)
        output = output.view(-1) #reshaping the tensor while being unsure of no of rows
        
        loss = lossfunc(output,y)
        running_loss+=loss.item()
        train_loss+=loss.item() # TENSOR TO SCALAR

        loss.backward()
        optimizer.step()

        if (i+1)%printfreq==0:
            log.write("Epoch: %d\tBatch: %d\nRunning Loss: %.4f\n"%(e+1,i+1,running_loss/printfreq))
            running_loss=0.0
    
    log.write("\nEpoch %d Train Loss: %.4f\n"%((e+1),train_loss/len(train_loader)))
    print("KKKKK")
    
    log.write("Training Complete.")

# Training Accuracy

Accuracy=0.0
ipsize=0

for data in train_loader:
    x,y=data
    
    output=Model(x)
    output = output.view(-1)
    
    print(output.shape,y.shape,"ASDFGH")
    output=output.round() # ROUNDING OFF THE PROBABILITIES TO EITHER 0(<0.5) OR 1(>0.5)
    print(output,"zxcvbnm")
    comp=torch.eq(output,y).type(torch.FloatTensor)
    Accuracy+=comp.sum().item()
    ipsize+=len(y)
    print(Accuracy,"Accuracyyyyy")
    # print("qwertyuiopqwertyuiop")
log.write("Training Accuracy: %.4f\n"%(Accuracy/20))


# Testing Accuracy

Accuracy=0.0
ipsize=0

for data in test_loader:
    x,y=data
    output=Model(x)
    output = output.view(-1)
    
    output=output.round()
    comp=torch.eq(output,y).type(torch.FloatTensor)
    Accuracy+=comp.sum().item()
    ipsize+=len(y)
    print("after2")
log.write("Testing Accuracy: %.4f\n"%(Accuracy/3))

#SAVING THE MODEL 
torch.save(Model,'model.pt')
