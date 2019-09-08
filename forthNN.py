import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x=Variable(torch.randn(batch_n,input_data),requires_grad=False)
y=Variable(torch.randn(batch_n,output_data),requires_grad=False)
# models=torch.nn.Sequential(
#     torch.nn.Linear(input_data,hidden_layer),
#     torch.nn.ReLU(),
#     torch.nn.Linear(hidden_layer,output_data)
# )
from collections import OrderedDict
models=torch.nn.Sequential(OrderedDict([
    ("Line1",torch.nn.Linear(input_data,hidden_layer)),
    ("Relu1",torch.nn.ReLU()),
    ("Line2",torch.nn.Linear(hidden_layer,output_data))
])
)
print(models)
epoch_n=10000
learning_rate=1e-4
loss_fn=torch.nn.MSELoss()
optimzer=torch.optim.Adam(models.parameters(),lr=learning_rate)#使用优化器优化模型的参数
for epoch in range(epoch_n):
    y_pred=models(x)
    loss=loss_fn(y_pred,y)
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data[0]))
    optimzer.zero_grad()
    #models.zero_grad()
    loss.backward()
    optimzer.step()
    # for param in models.parameters():
    #     param.data-=param.grad.data*learning_rate

