import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x=Variable(torch.randn(batch_n,input_data),requires_grad=False)#100x1000
y=Variable(torch.randn(batch_n,output_data),requires_grad=False)
w1=Variable(torch.randn(input_data,hidden_layer),requires_grad=True)#1000x100
w2=Variable(torch.randn(hidden_layer,output_data),requires_grad=True)#100x10
epoch_n=20
learning_rate=1e-6
for epoch in range(epoch_n):
    y_pred=x.mm(w1).clamp(min=0).mm(w2)
    loss=(y_pred-y).pow(2).sum()
    print("Epoch:{},Loss:{:.4f}".format(epoch,loss.data[0]))
    loss.backward()
    w1.data-=learning_rate*w1.grad.data
    w2.data-=learning_rate*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
