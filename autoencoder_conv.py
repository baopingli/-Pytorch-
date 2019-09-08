import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.image as mp

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
dataset_train=datasets.MNIST(root="./data",transform=transform,train=True,download=True)
dataset_test=datasets.MNIST(root="./data",transform=transform,train=False)
train_load=torch.utils.data.DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
test_load=torch.utils.data.DataLoader(dataset=dataset_test,batch_size=64,shuffle=True)
images,label=next(iter(train_load))
print(images.numpy().shape)
images_example=torchvision.utils.make_grid(images)
images_example=images_example.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
images_example=images_example*std+mean
mp.imsave("bb.png",images_example)
noisy_images=images_example+0.5*np.random.randn(*images_example.shape)
noisy_images=np.clip(noisy_images,0.,1.)
mp.imsave("bb1.png",noisy_images)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.decoder=torch.nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,input):
        output=self.encoder(input)
        output=self.decoder(output)
        return output

model=AutoEncoder()
print(model)
Use_gpu=torch.cuda.is_available()
if Use_gpu:
    model=model.cuda()
print(model)
optimizer=torch.optim.Adam(model.parameters())
loss_f=torch.nn.MSELoss()
epoch_n=5
for epoch in range(epoch_n):
    running_loss=0.0
    print("Epoch {}/{}".format(epoch,epoch_n))
    print("-"*10)
    for data in train_load:
        X_train,_=data
        noisy_X_train=X_train+0.5*torch.randn(X_train.numpy().shape)
        noisy_X_train=torch.clamp(noisy_X_train,0.,1.)
        X_train,noisy_X_train=Variable(X_train.cuda()),Variable(noisy_X_train.cuda())
        train_pre=model(noisy_X_train)
        loss=loss_f(train_pre,X_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.data[0]
print("Loss is :{:.4f}".format(running_loss/len(dataset_train)))


