import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)
#一般时候是False
#variable是篮子，会搭建计算图纸，计算图纸搭建的时候进行误差反向传播，通过variable
#false,不会计算，神经网络计算节点要计算
print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out)

v_out.backward()
#模拟v_out反向传递
#v_out = 1/4 * sum(variable * variable)
#d(v_out)/d(variable) = 1/4 * 2 * variable = 1/2 * variable


print(variable.grad)
#反向传递后更新值

print(variable)
print(variable.data)
print(variable.data.numpy())

#fake data
x = torch.linspace(-5,5,200)
x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

