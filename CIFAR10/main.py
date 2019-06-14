'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--GPU',help="whether use GPU")

    args=parser.parse_args()


    if args.GPU == 'True' and torch.cuda.is_available():
        use_GPU = True
    else:
        use_GPU = False

    print("USE GPU:", use_GPU)


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()


    if use_GPU:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)




    for epoch in range(200):
        for step, (b_x, b_y) in enumerate(trainloader):
            if use_GPU:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            output = net(b_x)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            correct = 0
            total = 0
            for step, (b_x, b_y) in enumerate(testloader):
                if use_GPU:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                testoutput = net(b_x)

                if use_GPU:
                    pre_y = torch.max(testoutput, 1)[1].cuda().data.squeeze()
                else:
                    pre_y = torch.max(testoutput, 1)[1].data.squeeze()

                right = torch.sum(pre_y==b_y).type(torch.FloatTensor)
                total += b_y.size()[0]
                correct += right
            
            print("Epoch {}, Accuracy:{}".format(epoch, correct/total))

    print("Finish Training")