import torch
import torchvision
from torchvision import datasets,transforms
import os
import matplotlib.pyplot as plt
import time
import matplotlib.image as mp
from torch.autograd import Variable

data_dir="./DogsVSCats/"
data_transform={x:transforms.Compose([transforms.Scale([64,64]),
                                      transforms.ToTensor()])
                for x in ["train","valid"]}
image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_transform[x])
                for x in ["train","valid"]}
dataloader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                          batch_size=16,
                                          shuffle=True)
            for x in ["train","valid"]}
X_example,Y_example=next(iter(dataloader["train"]))
print(u"X_example个数{}".format(len(X_example)))
print(u"Y_example个数{}".format(len(Y_example)))
index_classes=image_datasets["train"].class_to_idx
print(index_classes)
example_clasees=image_datasets["train"].classes
print(example_clasees)
img=torchvision.utils.make_grid(X_example)
img=img.numpy().transpose([1,2,0])
print([example_clasees[i] for i in Y_example])
mp.imsave('show_examples.png',img)
#plt.imshow(img)
#plt.show()
#
class Models(torch.nn.Module):
    def __init__(self):
        super(Models,self).__init__()
        self.Conv=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),

            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),

            torch.nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.Classes=torch.nn.Sequential(
            torch.nn.Linear(4*4*512,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,2)
        )
    def forward(self,input):
        x=self.Conv(input)
        x=x.view(-1,4*4*512)
        x=self.Classes(x)
        return x
model=Models()
print(model)
#
loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.00001)

print(torch.cuda.is_available())
Use_gpu=torch.cuda.is_available()
if Use_gpu:
    model=model.cuda()

epoch_n=10
time_open=time.time()
for epoch in range(epoch_n):
    print("Epoch:{}/{}".format(epoch,epoch_n-1))
    print("-"*10)
    for phase in ["train","valid"]:
        if phase=="train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)
        running_loss=0.0
        running_corrects=0
        for batch,data in enumerate(dataloader[phase],1):
            X,y=data
            if Use_gpu:
                X,y=Variable(X.cuda()),Variable(y.cuda())
            else:
                X,y=Variable(X),Variable(y)
            y_pred=model(X)
            _,pred=torch.max(y_pred.data,1)
            optimizer.zero_grad()
            loss=loss_f(y_pred,y)
            if phase=="train":
                loss.backward()
                optimizer.step()
            #print(loss.data.item())
            #running_loss+=loss.data[0]
            running_loss += loss.data.item()
            running_corrects+=torch.sum(pred==y.data)
            if batch%500==0 and phase=="train":
                print("Batch {},Train Loss:{:.4f},Train ACC:{:.4f}".format(batch,running_loss/batch,100*running_corrects/(16*batch)))
        epoch_loss=running_loss*16/len(image_datasets[phase])
        epoch_acc=100*running_corrects/len(image_datasets[phase])
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))
time_end=time.time()-time_open
print(time_end)