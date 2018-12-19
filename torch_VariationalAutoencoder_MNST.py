import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import visdom
import time
import numpy as np
import torchvision
import matplotlib.pyplot as plt
# https://blog.csdn.net/tonydz0523/article/details/79084192

BATCH_SIZE = 64
LR = 0.001

hidden_size = 3

train_dataset = torchvision.datasets.MNIST(
    root='./mnist/',        # 保存或者提取位置
    train = True,       # this is training data
    transform = torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                                                    #0-255 到 0-1
    download=False,
)

test_dataset = torchvision.datasets.MNIST(
    root='./mnist/',        # 保存或者提取位置
    train = False,       # this is training data
    transform = torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                                                    #0-255 到 0-1
)


train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, 400, False)

dataiter = iter(test_loader)
inputs, labels = dataiter.next()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))

        self.fc_encode1 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_encode2 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_decode = nn.Linear(hidden_size, 16 * 7 * 7)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, 2, 1),
                                     nn.Sigmoid())

    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out)
        return self.fc_encode1(out.view(out.size(0), -1)), self.fc_encode2(out.view(out.size(0), -1))

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        return eps.mul(var).add_(mean)

    def decoder(self, x):
        out = self.fc_decode(x)
        out = self.deconv1(out.view(x.size(0), 16, 7, 7))
        out = self.deconv2(out)
        return out

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, code, mean, std

net = VAE()

bce = nn.BCELoss()
bce.size_average = False
data = torch.Tensor(BATCH_SIZE ,28*28)

def loss_f(out, target, mean, std):
    bceloss = bce(out, target)
    latent_loss= torch.sum(mean.pow(2).add_(std.exp()).mul_(-1).add_(1).add_(std)).mul_(-0.5)
    return bceloss + latent_loss

optimizer = torch.optim.Adam(net.parameters(), lr=LR)


for epoch in range(30):
    net.train()
    for step, (images, _) in enumerate(train_loader, 1):
        net.zero_grad()
        data.data.resize_(images.size()).copy_(images)
        output, _, mean, std = net(data)
        loss = loss_f(output, data, mean, std)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            net.eval()
            eps = Variable(inputs)
            output= net(eps)[0]

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                       100. * step / len(train_loader),
                       loss.item()/BATCH_SIZE))


            if step == 200:
                plt.imshow(output[0].detach().numpy()[0], cmap='gray')
                plt.title("epoch:{}".format(epoch))
                # plt.show()
                plt.savefig('fig_tem/epoch{}.png'.format(epoch))

if hidden_size == 3:
    for step, (images, labels) in enumerate(test_loader, 1):
        if step > 1:
            break
        mean, std = net.encoder(images)
        tags = net.sampler(mean, std)

