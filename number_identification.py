import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mp


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

data_train=datasets.MNIST(root="./data/",
                          transform=transform,
                          train=True,
                          download=True)
data_test=datasets.MNIST(root="./data/",
                         transform=transform,
                         train=False)

data_loader_train=torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=64,
                                              shuffle=True)
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=64,
                                             shuffle=True)
images,labels=next(iter(data_loader_train))
img=torchvision.utils.make_grid(images)
img=img.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
print([labels[i] for i in range(64)])
# plt.imshow(img)
# plt.savefig('./img.jpg')
mp.imsave('res.png',img)
print('save res.png')

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))
    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x

model=Model()
cost=torch.nn.CrossEntropyLoss()
Use_gpu=torch.cuda.is_available()
if Use_gpu:
    model=model.cuda()
optimizer=torch.optim.Adam(model.parameters())
print(model)

n_epochs=5
for epoch in range(n_epochs):
    running_loss=0.0
    running_correct=0
    print("Epoch:{}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in data_loader_train:
        X_train,Y_train=data
        X_train,Y_train=Variable(X_train.cuda()),Variable(Y_train.cuda())
        outputs=model(X_train)
        _,pred=torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss=cost(outputs,Y_train)
        loss.backward()
        optimizer.step()
        running_loss+=loss.data[0]
        running_correct+=torch.sum(pred==Y_train.data)
    testing_correct=0
    for data in data_loader_test:
        X_test,Y_test=data
        X_test,Y_test=Variable(X_test.cuda()),Variable(Y_test.cuda())
        outputs=model(X_test)
        _,pred=torch.max(outputs.data,1)
        testing_correct+=torch.sum(pred==Y_test.data)
    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is :{:.4f}".format(
        running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)
    ))
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=4,shuffle=True)
X_test,Y_test=next(iter(data_loader_test))
# X_test,Y_test=Variable(X_test.cuda()),Variable(Y_test.cuda())
inputs=Variable(X_test)
pred=model(inputs)
_,pred=torch.max(pred,1)
print("Predict Label is:",[i for i in pred.data])
print("Real Label is:",[i for i in Y_test])
img=torchvision.utils.make_grid(X_test)
img=img.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
mp.imsave('res1.png',img)
print("save res1.png")

