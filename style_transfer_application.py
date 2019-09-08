import torch
import torchvision
from torchvision import transforms,models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

#首先定义内容度量和风格度量
#引入模型
#添加需要的模块
#训练

transform=transforms.Compose([transforms.Scale([224,224]),
                               transforms.ToTensor()])
def loadimg(path=None):
    img=Image.open(path)
    img=transform(img)
    img=img.unsqueeze(0)
    return img
content_img=loadimg("./images/4.jpg")
content_img=Variable(content_img).cuda()
style_img=loadimg("./images/1.jpg")
style_img=Variable(style_img).cuda()

class Content_loss(torch.nn.Module):
    def __init__(self,weight,target):
        super(Content_Loss,self).__init__()
        self.weight=weight
        self.target=target.detach()*weight
        self.loss_fn=torch.nn.MSELoss()
    def forward(self,input):
        self.loss=self.loss_fn(input*self.weight, self.target)
        return input
    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
class Gram_matrix(torch.nn.Module):
    def forward(self,input):
        a,b,c,d=input.size()
        feature=input.view(a*b,c*d)
        gram=torch.mm(feature,feature.t())
        return gram.div(a*b*c*d)
class Style_loss(torch.nn.Module):
    def __index__(self,weight,target):
        super(Style_loss,self).__init__()
        self.weight=weight
        self.target=target.detach()*weight
        self.loss_fn=torch.nn.MSELoss()
        self.gram=Gram_matrix()
    def forward(self,input):
        self.Gram=self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss=self.loss_fn(self.Gram,self.target)
        return input
    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
use_gpu=torch.cuda.is_available()
cnn=models.vgg16(pretrained=True).features
if use_gpu:
    cnn=cnn.cuda()
model=copy.deepcopy(cnn)
content_layer=["Conv_3"]#分别制定了哪一层提取
style_layer=["Conv_1","Conv_2","Conv_3","Conv_4"]
content_losses=[]
style_losses=[]
conten_weight=1
style_weight=1000
new_model=torch.nn.Sequential()#新建一个空的model
model=copy.deepcopy(cnn)
gram=Gram_matrix()
if use_gpu:
    new_model=new_model.cuda()
    gram=gram.cuda()
index=1
for layer in list(model)[:8]:#将vgg16的前8层搞到手
    if isinstance(layer,torch.nn.Conv2d):
        name="Conv_"+str(index)
        new_model.add_module(name,layer)
        if name in content_layer:
            target=new_model(content_img).clone()
            content_loss=Content_loss(conten_weight,target)
            new_model.add_module("content_loss_"+str(index),content_loss)
            content_losses.append(content_loss)
        if name in style_layer:
            target=new_model(style_img).clone()
            target=gram(target)
            style_loss=Style_loss(style_weight,target)
            new_model.add_module("style_loss_"+str(index),style_loss)
            style_losses.append(style_loss)
    if isinstance(layer,torch.nn.ReLU):
        name="Relu_"+str(index)
        new_model.add_module(name,layer)
        index=index+1
    if isinstance(layer,torch.nn.MaxPool2d):
        name="MaxPool_"+str(index)
        new_model.add_module(name,layer)
input_img=content_img.clone()
parameter=torch.nn.Parameter(input_img.data)
optimizer=torch.optim.LBFGS([parameter])
epoch_n=300
epoch=[0]
while epoch[0]<=epoch_n:
    def closure():
        optimizer.zero_grad()
        style_score=0
        content_score=0
        parameter.data.clamp_(0,1)
        new_model(parameter)
        for sl in style_losses:
            style_score+=sl.backward()
        for cl in content_losses:
            content_loss+=cl.backward()
        epoch[0]+=1
        if epoch[0]%50==0:
            print("Epoch:{} Style Loss:{:.4f} Content Loss:{:.4f}".format(epoch[0],style_score.data[0],content_score.data[0]))
        return style_score+content_score
    optimizer.step(closure)

