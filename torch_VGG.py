import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

import torchvision
import matplotlib.pyplot as plt
import os

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

#Minist
train_data = torchvision.datasets.MNIST(
    root='./mnist/',        # 保存或者提取位置
    train = True,       # this is training data
    transform = torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                                                    #0-255 到 0-1
    download=False,
)


#plot one example
print(train_data.train_data.size())                  # (60000, 28, 28)
print(train_data.train_labels.size())                 # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

#
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height(灰度=1， RGB = 3）
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)

            nn.ReLU(),  # activation

            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),  # (32,14,14)
            nn.MaxPool2d(2), # (32,7,7)
        )

        self.conv3 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),  # (64,7,7)
            nn.MaxPool2d(2),  # (64,3,3)

        )

        self.out = nn.Linear(64 * 3 * 3, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32,7,7)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x



cnn = VGG()
print(cnn)
#net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()                # the target label is not one-hotted


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]                          # cnn output
        loss = loss_func(output, b_y)               # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' %loss.data.numpy(), ' | test accuracy: %.2f' % accuracy)



# print 10 prediction form test dataset
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')