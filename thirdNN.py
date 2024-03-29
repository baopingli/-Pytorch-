import torch
from torch.autograd import Variable
batch_n=64
hidden_layer=100
input_data=1000
output_data=10
class Model(torch.nn.Module):
    def forward(self,input,w1,w2):
        x=torch.mm(input,w1)
        x=torch.clamp(x,min=0)
        x=torch.mm(x,w2)
        return x
    def backward(self):
        pass
model=Model()
x=Variable(torch.randn(batch_n,input_data),requires_grad=False)
y=Variable(torch.randn(batch_n,output_data),requires_grad=False)
w1=Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2=Variable(torch.randn(hidden_layer,output_data),requires_grad=True)
epoch_n=30
learning_rate=1e-6
for epoch in range(epoch_n):
    y_pred=model(x,w1,w2)
    loss=(y_pred-y).pow(2).sum()
    print("Epoch:{},loss:{:.4f}".format(epoch,loss.data[0]))
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
