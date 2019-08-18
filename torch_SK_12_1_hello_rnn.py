import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [0,1,0,2,3,3]
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]
                  ]
y_data = [1, 0, 2, 3, 3, 4]
# ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

# As we have one batch of samples, we will change them to variables only once
inputs = torch.Tensor(x_one_hot)
labels = torch.LongTensor(y_data)

num_classes = 5
input_size = 5
hidden_size = 5
batch_size = 1
sequence_length = 1
num_layers = 1

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    def forward(self, hidden, x):
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))



# Instantiate RNN model
model = Model()
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    sys.stdout.write("predicted string: ")
    # 使用sys.stdout.write， 相比于print， 其不会换行
    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        hidden, output = model(hidden, input)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        loss += loss_func(output, label.unsqueeze(0))

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))

    loss.backward()
    optimizer.step()

print("finished")
