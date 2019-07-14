import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(28*28, 30),
            nn.Tanh(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(30, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x


trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([.5], [.5]),
    ]
)

BATCHSIZE=100
DOWNLOAD_MNIST=False
EPOCHES=200
LR=0.001

train_data=torchvision.datasets.MNIST(
    root="./mnist",train=True,transform=trans,download=DOWNLOAD_MNIST,
)
test_data=torchvision.datasets.MNIST(
    root="./mnist",train=False,transform=trans,download=DOWNLOAD_MNIST,
)
train_loader=DataLoader(train_data,batch_size=BATCHSIZE,shuffle=True)
test_loader =DataLoader(test_data,batch_size=BATCHSIZE,shuffle=False)



mlps=[MLP() for i in range(10)]
optimizer=torch.optim.Adam([{"params":mlp.parameters()} for mlp in mlps],lr=LR)
loss_function=nn.CrossEntropyLoss()


#
for ep in range(EPOCHES):
    for img, label in train_loader:
        optimizer.zero_grad()
        for mlp in mlps:
            out = mlp(img)
            loss = loss_function(out, label)
            loss.backward()
        optimizer.step()

    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]

    for img, label in test_loader:
        for i, mlp in enumerate(mlps):
            out = mlp(img)
            _, prediction = torch.max(out, 1)
            pre_num = prediction.numpy()
            mlps_correct[i] += (pre_num == label.numpy()).sum()

            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(BATCHSIZE)]

        vote_correct += (result == label.numpy()).sum()

    print("epoch:" + str(ep)+"总的正确率"+str(vote_correct/len(test_data)))

    for idx, correct in enumerate(mlps_correct):
        print("网络" + str(idx) + "的正确率为：" + str(correct / len(test_data)))





