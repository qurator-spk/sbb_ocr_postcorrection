import torch
import torch.nn as nn


class DetectorLSTM(nn.Module):
    '''The Detector model (based on an LSTM architecture).'''

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

        super(detectorLSTM, self).__init__()

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


class DetectorGRU(nn.Module):
    '''The Detector model (based on a GRU architecture).'''

    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False, dropout=0, activation='softmax', device='cpu'):
        '''
        Keyword arguments:
        input_size (int) -- the size of the input vector
        hidden_size (int) -- the size of the GRU's hidden node
        output_size (int) -- the size of the output vector
        batch_size (int) -- the batch size
        num_layers (int) -- the number of GRU layers (default: 1)
        bidirectional (bool) -- declares if model is trained bidirectionally
                                (default: False)
        dropout (float) -- the dropout probability (default: 0)
        activation (str) -- the activation function (final layer)
                            (default: softmax)
        device (str) -- the device used for training (default: 'cpu')
        '''

        super(detectorGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.node_type = 'gru'
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size*2, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

        if activation == 'softmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise('Activation function needs to be softmax.')

    def forward(self, x, hidden):

        embedded = self.embedding(x).view(self.batch_size, 1, -1)
        gru_output, hidden = self.gru(embedded, hidden)
        linear_output = self.linear(gru_output).view(self.batch_size, -1)

        return self.activation(linear_output), gru_output, hidden

    def init_hidden_state(self):
        if self.bidirectional:
            return torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)


class DetectorCNN(nn.Module):
    '''
    The Detector model (based on a CNN architecture).

    !!!TO BE IMPLEMENTED!!!
    '''

    def __init__(self, seq):
        super(detectorCNN, self).__init__()
        pass
