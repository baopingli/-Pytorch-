import torch
# a=torch.FloatTensor(2,3)
# b=torch.FloatTensor([2,3,4,5])
# c=torch.IntTensor(2,3)
# d=torch.IntTensor([2,3,4,5])
# e=torch.randn(2,3)
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# a=torch.range(1,20,2)
#import torch
# a=torch.randn(2,3)
# print(a)
# b=torch.clamp(a,-0.1,0.1)
# print(b)
# a=torch.randn(2,3)
# print(a)
# b=torch.randn(3)
# print(b)
# c=torch.mv(a,b)
# a=torch.randn(2,2)
# print(a)
# b=a.clone()
# print(b)
a=torch.randn(2,3)
b=torch.randn(3,2)
c=a.mm(b)
print(c)