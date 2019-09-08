import torch
batch_n=100
hidden_layer=100
input_data=1000
output_data=10
x=torch.randn(batch_n,input_data)#100x1000
y=torch.randn(batch_n,output_data)
w1=torch.randn(input_data,hidden_layer)#1000x100
w2=torch.randn(hidden_layer,output_data)
a=torch.randn(2,3)
b=torch.randn(3,2)
epoch_n=20
learning_rate=1e-6
for epoch in range(epoch_n):
    h1=x.mm(w1)
    h1=h1.clamp(min=0)
    y_pred=h1.mm(w2)#100*10
    loss=(y_pred-y).pow(2).sum()
    print("Epoch:{},Loss:{:.4f}".format(epoch,loss))
    grad_y_pred=2*(y_pred-y)
    grad_w2=h1.t().mm(grad_y_pred)
    grad_h=grad_y_pred.clone()
    grad_h=grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1=x.t().mm(grad_h)
    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2

