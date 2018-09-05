import torch
import torch.nn as nn
from torch.autograd import Variable

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
y_data = [1, 0, 2, 3, 3, 4]    # ihello

inputs = Variable(torch.LongTensor(x_data))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5
embedding_size = 10  # embedding size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embeddinng = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=5, batch_first=True)

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        ))

        emb = self.embeddinng(x)
        emb = emb.view(batch_size, sequence_length, -1)

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)

        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))

# Instantiate RNN model
model = Model()
print(model)

loss_func = torch.nn.CrossEntropyLoss()
optimizer  =torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")




