import torch
from torch.autograd import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def SeriesGen(N):
    x = torch.arange(1, N, 0.01)
    return torch.sin(x)

def trainDataGen(seq, k):
    dat = list()
    L = len(seq)
    for i in range(L-k-1):
        indat = seq[i:i+k]
        outdata = seq[i+1:i+k+1]
        dat.append((indat, outdata))
    return dat

# def ToVariable(x):
#     tmp = torch.FloatTensor(x)
#     return Variable(tmp)

y = SeriesGen(10)
dat = trainDataGen(y.numpy(), 10)

class LSTMpred(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1,self.hidden_dim))

    def forward(self, seq):

        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden
        )

        outdat = self.hidden2out(lstm_out.view(len(seq), -1)).squeeze(1)

        return outdat

model = LSTMpred(1, 6)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(15):
    print(epoch)
    for seq, outs in dat[:700]:
        seq = torch.FloatTensor(seq)
        outs = torch.FloatTensor(outs)
        # print(seq)
        optimizer.zero_grad()

        model.hidden = model.init_hidden()

        modout = model(seq)
        # print('modout',modout)
        # print(outs)
        loss = loss_function(modout, outs)
        loss.backward()
        optimizer.step()


predDat = []
for seq, trueVal in dat[700:]:
    seq = torch.FloatTensor(seq)
    outs = torch.FloatTensor(outs)
    predDat.append(model(seq)[-1].data.numpy())

fig = plt.figure()
plt.plot(y.numpy())
plt.plot(range(700, 889), predDat)
plt.show()