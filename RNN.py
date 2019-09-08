import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
dataset_train=datasets.MNIST(root="./data",
                             transform=transform,
                             train=True,
                             download=True)
dataset_test=datasets.MNIST(root="./data",
                            transform=transform,
                            train=False)
train_load=torch.utils.data.DataLoader(dataset=dataset_train,
                                       batch_size=64,
                                       shuffle=True)
test_load=torch.utils.data.DataLoader(dataset=dataset_test,
                                      batch_size=64,
                                      shuffle=True)
images,label=next(iter(train_load))

images_example=torchvision.utils.make_grid(images)
images_example=images_example.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
images_example=images_example*std+mean
# plt.imshow(images_example)
# plt.show()
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=torch.nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.output=torch.nn.Linear(128,10)
    def forward(self,input):
        output,_=self.rnn(input,None)
        output=self.output(output[:,-1,:])
        return output
model=RNN().cuda()
optimizer=torch.optim.Adam(model.parameters())
loss_f=torch.nn.CrossEntropyLoss()
epoch_n=10
for epoch in range(epoch_n):
    running_loss=0.0
    running_correct=0
    testing_correct=0
    print("Epoch {}/{}".format(epoch,epoch_n))
    print("-"*10)
    for data in train_load:
        X_train,y_train=data
        X_train=X_train.view(-1,28,28)
        X_train,y_train=Variable(X_train.cuda()),Variable(y_train.cuda())
        y_pred=model(X_train)
        loss=loss_f(y_pred,y_train)
        _,pred=torch.max(y_pred.data,1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.data[0]
        running_correct+=torch.sum(pred==y_train.data)
    for data in test_load:
        X_test,y_test=data
        X_test=X_test.view(-1,28,28)
        X_test,y_test=Variable(X_test.cuda()),Variable(y_test.cuda())
        outputs=model(X_test)
        _,pred=torch.max(outputs.data,1)
        testing_correct+=torch.sum(pred==y_test.data)
print("Loss is:{:.4f},Train Accuracy is :{:.4f}%,Test Accuracy is :{:.4f}".format(running_loss/len(dataset_train),100*running_correct/len(dataset_train),100*testing_correct/len(dataset_test)))
