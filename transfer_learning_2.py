import torch
import torchvision
from  torchvision import datasets,models,transforms
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

data_dir="./DogsVSCats/"
Use_gpu=True
data_transform={x:transforms.Compose([transforms.Scale([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
                for x in ["train","valid"]}
image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_transform[x])
                for x in ["train","valid"]}
dataloader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                          batch_size=16,
                                          shuffle=True)
            for x in ["train","valid"]}
X_example,y_example=next(iter(dataloader["train"]))
example_clasees=image_datasets["train"].classes
index_classes=image_datasets["train"].class_to_idx

model=models.vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad=False
model.classifier=torch.nn.Sequential(torch.nn.Linear(25088,4096),
                                     torch.nn.ReLU(),
                                     torch.nn.Dropout(p=0.5),
                                     torch.nn.Linear(4096,4096),
                                     torch.nn.ReLU(),
                                     torch.nn.Dropout(p=0.5),
                                     torch.nn.Linear(4096,2))
if Use_gpu:
    model=model.cuda()
cost=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.classifier.parameters())
loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.classifier.parameters(),lr=0.00001)
epoch_n=5
time_open=time.time()
for epoch in range(epoch_n):
    print("Epoch{}/{}".format(epoch,epoch_n-1))
    print("-"*10)
    for phase in ["train","valid"]:
        print("Epoch{}/{}".format(epoch,epoch_n-1))
        print ("-"*10)
        for phrase in ["train","valid"]:
            if phase =="train":
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
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss+=loss.data.item()
                running_corrects+=torch.sum(pred==y.data)
                if batch%500 == 0 and phase == "train":
                    print("Batch{},Train Loss:{:.4f},Train ACC:{:.4f}".format(
                        batch,running_loss/batch,100*running_corrects/(16*batch)
                    ))
            epoch_loss=running_loss*16/len(image_datasets[phase])
            epoch_acc=100*running_corrects/len(image_datasets[phase])
            print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))
time_end=time.time()-time_open
print(time_end)

