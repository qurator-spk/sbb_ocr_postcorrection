import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):

        x = self.embedding(x)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False, dropout=0, activation='softmax', device='cpu'):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        if activation == 'softmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise('Activation function needs to be "softmax".')

    def forward(self, x, hidden, cell):

        embedded = self.embedding(x).view(self.batch_size, 1, -1)
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # not sure; maybe wrong dimensions
        linear_output = self.fc(lstm_output).view(self.batch_size, -1)

        return self.activation(linear_output), hidden, cell

    def init_hidden_state(self):
        if self.bidirectional:
            return torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)

    def init_cell_state(self):
        if self.bidirectional:
            return torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)

def real_loss(d_logits, criterion, smooth=False):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape

    if smooth:
        labels = torch.ones(size)*0.9
    else:
        labels = torch.ones(size)

    loss = criterion(d_logits, labels)
    return loss

def fake_loss(d_logits, criterion):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape

    labels = torch.zeros(size)

    loss = criterion(d_logits, labels)
    return loss
