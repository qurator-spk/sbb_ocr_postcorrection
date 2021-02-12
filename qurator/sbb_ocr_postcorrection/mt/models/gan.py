import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0):
        super(DiscriminatorLinear, self).__init__()

        self.model_type = 'linear'

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

class DiscriminatorLSTM(nn.Module):
    '''The Discriminator model (based on an LSTM architecture).'''

    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False, dropout=0, activation='softmax', device='cpu'):
        '''
        Keyword arguments:
        input_size (int) -- the size of the input vector
        hidden_size (int) -- the size of the LSTM's hidden node
        output_size (int) -- the size of the output vector
        batch_size (int) -- the batch size
        num_layers (int) -- the number of LSTM layers (default: 1)
        bidirectional (bool) -- declares if model is trained bidirectionally
                                (default: False)
        dropout (float) -- the dropout probability (default: 0)
        activation (str) -- the activation function (final layer)
                            (default: softmax)
        device (str) -- the device used for training (default: 'cpu')
        '''

        super(DiscriminatorLSTM, self).__init__()

        self.model_type = 'lstm'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = hidden_size
        self.batch_size = batch_size
        self.node_type = 'lstm'
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size*2, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

        if activation == 'softmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise('Activation function needs to be "softmax".')

    def forward(self, x, hidden, cell):

        embedded = self.embedding(x).view(self.batch_size, 1, -1)
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        linear_output = self.linear(lstm_output).view(self.batch_size, -1)

        return self.activation(linear_output), lstm_output, hidden, cell

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

class GeneratorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False, dropout=0, activation='softmax', device='cpu'):
        super(GeneratorLSTM, self).__init__()

        self.model_type = 'lstm'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
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

def real_loss(d_logits, criterion, smooth=False, device='cpu'):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape[0]

    if smooth:
        labels = torch.ones(size, device=device)*0.9
    else:
        labels = torch.ones(size, device=device, dtype=torch.long)

    loss = criterion(d_logits, labels)
    return loss

def fake_loss(d_logits, criterion, device='cpu'):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape[0]

    labels = torch.zeros(size, device=device, dtype=torch.long)

    loss = criterion(d_logits, labels)
    return loss
