import torch
import numpy as np
from torch.autograd import Variable

# x = torch.Tensor(5,3)
# # x未初始化
# y = torch.rand(5,3)
# # y随机初始化
# print('x',x)
# print('y',y)
# print(x.size)
#
# z = torch.rand(5,3)
# t2 = y + z
# t1 = torch.add(z,y)
# print('t1',t1)
# print('t2',t2)
#
# t3 = y.add_(z)
# print('t3',t3)
# # 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# # 例如：x.copy_(y), x.t_(), 这俩都会改变x的值
#
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print('b',b)
#
# c = np.ones(5)
# d = torch.from_numpy(c)
# np.add(c,1,out=c)
# print('c',c)
# print('d',d)
#
# tem = Variable(torch.ones(2,2),requires_grad = True)
# tem2 = tem + 2
# # print(tem2.creator)
# tem3 = tem2 * tem2 * 3
# out = tem3.mean()
#
# print(out)
# print(out.backward())
# print(tem.grad)

np_Data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_Data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:',np_Data,
    '\ntorch',torch_data,
    '\ntensor to array',tensor2array
)

#abs
data = [-2,-1,-3,1]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy:',np.abs(data),
    '\ntorch:',torch.abs(tensor)
)

#sin
print(
    '\sin',
    '\nnumpy',np.sin(data),
    '\ntorch',torch.sin(tensor)
)

#mean
print(
    '\mean',
    '\nnumpy',np.mean(data),
    '\ntorch',torch.mean(tensor)
)

data1 = [[1,2],[3,4]]
tensor1 = torch.FloatTensor(data1)
data = np.array(data1)
#correnct
print(
    '\nmatrix multiplication',
    '\numpy1',np.matmul(data,data),
    '\ntorch1',torch.mm(tensor1,tensor1),
    # '\numpy2',np.dot(data),
    # '\ntorch2',torch.dot(tensor1),
    # '\ntorch2', torch.dot(tensor, tensor),

)