import torch
import torch.nn as nn
import torch.nn.functional as F


#####################
#                   #
# ENCODER COMPONENT #
#                   #
#####################

class EncoderLSTM(nn.Module):
    '''The Encoder component (based on an LSTM architecture).'''

    def __init__(self, input_size, hidden_size, batch_size, num_layers=1, dropout=0, device='cpu'):
        '''
        Keyword arguments:
        input_size (int) -- the size of the input vector
        hidden_size (int) -- the size of the LSTM's hidden node
        batch_size (int)-- the batch size
        num_layers (int) -- the number of LSTM layers (default: 1)
        dropout (float) -- the dropout probability (default: 0)
        device (str) -- the device used for training (default: 'cpu')
        '''
        super(EncoderLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden, cell):

        if len(x.shape) > 1:
            embedded = x.view(self.batch_size, 1, -1)
        else:
            embedded = self.embedding(x).view(self.batch_size, 1, -1)
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        return output, hidden, cell

    def init_hidden_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

    def init_cell_state(self):
        return torch.zeros(1, self.batch_size, self.cell_size, device=self.device)


class EncoderGRU(nn.Module):
    '''The Encoder component (based on a GRU architecture).'''

    def __init__(self, input_size, hidden_size, batch_size, num_layers=1, dropout=0, device='cpu'):
        '''
        Keyword arguments:
        input_size (int) -- the size of the input vector
        hidden_size (int) -- the size of the GRU's hidden node
        batch_size (int)-- the batch size
        num_layers (int) -- the number of GRU layers (default: 1)
        dropout (float) -- the dropout probability (default: 0)
        device (str) -- the device used for training (default: 'cpu')
        '''
        super(EncoderGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(self.batch_size, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


#####################
#                   #
# DECODER COMPONENT #
#                   #
#####################

class DecoderLSTM(nn.Module):
    '''The Decoder component (based on an LSTM architecture; no attention).'''

    def __init__(self, hidden_size, output_size, batch_size, num_layers=1, dropout=0, device='cpu'):
        '''
        Keyword arguments:
        hidden_size (int) -- the size of the LSTM's hidden node
        output_size (int) -- the size of the output vector
        batch_size (int)-- the batch size
        num_layers (int) -- the number of LSTM layers (default: 1)
        dropout (float) -- the dropout probability (default: 0)
        device (str) -- the device used for training (default: 'cpu')
        '''
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.cell_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, cell):

        output = self.embedding(x).view(self.batch_size, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = self.fc(output).view(self.batch_size, -1)
        output = F.log_softmax(output, dim=1)

        #output = self.softmax(self.fc(output)).view(self.batch_size, -1)
        return output, hidden, cell

    def init_hidden_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

    def init_cell_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class DecoderGRU(nn.Module):
    '''
    The Decoder component (based on a GRU architecture; no attention).

    !!!UNTESTED!!!

    '''

    def __init__(self, hidden_size, output_size, batch_size, device='cpu'):
        super(DecoderGRU, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output = self.embedding(x).view(self.batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class AttnDecoderLSTM(nn.Module):
    '''The Decoder component (based on an LSTM architecture; with attention).'''

    def __init__(self, hidden_size, output_size, batch_size, seq_length, num_layers=1, dropout=0, device='cpu'):
        '''
        Keyword arguments:
        hidden_size (int) -- the size of the LSTM's hidden node
        output_size (int) -- the size of the output vector
        batch_size (int)-- the batch size
        seq_length (int) -- the length of the sequence
        num_layers (int) -- the number of LSTM layers (default: 1)
        dropout (float) -- the dropout probability (default: 0)
        device (str) -- the device used for training (default: 'cpu')
        '''

        super(AttnDecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.cell_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.seq_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, cell, encoder_outputs):

        # import pdb; pdb.set_trace()

        if len(x.shape) > 1:
            embedded = x.view(1, self.batch_size, -1)
        else:
            embedded = self.embedding(x).view(1, self.batch_size, -1)  # NOTE: Changed from .view(self.batch_size, 1, -1)
        
        embedded = F.dropout(embedded, p=self.dropout)

        ### Changes made for batch_size > 1; needs to be checked ##############

        #attn_weights = F.softmax(
        #    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                         encoder_outputs.unsqueeze(0))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        #output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = torch.cat((embedded[0], attn_applied.squeeze()), 1)
        output = self.attn_combine(output).unsqueeze(0)
        #######################################################################

        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, cell, attn_weights

    def dot_scoring_function(self):
        '''
        '''
        pass

    def general_scoring_function(self):
        '''
        '''
        pass

    def concat_scoring_function(self):
        '''
        '''
        pass

    def bahdanau_scoring_function(self):
        '''
        '''

    def init_hidden_state(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

    def init_cell_state(self):
        return torch.zeros(1, self.batch_size, self.cell_state, device=self.device)
