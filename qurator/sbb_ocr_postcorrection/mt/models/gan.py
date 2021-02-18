import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorCNN(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                filter_sizes=[3,4,5],
                num_filters=[100, 100, 100],
                num_classes = 1,                
                kernel_size=2, 
                stride=2, 
                padding=1, 
                lrelu_neg_slope=0.2,
                dropout_prob=0.5):
        super(DiscriminatorCNN, self).__init__()

        self.model_type = 'cnn'

        self.seq_len = 40

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.lrelu_neg_slope = lrelu_neg_slope

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.conv1_list = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

        #self.conv1 = nn.Conv1d(self.seq_len, self.seq_len*2 , kernel_size=kernel_size, stride=stride, padding=padding)
        #self.conv2 = nn.Conv1d(self.seq_len*2, self.seq_len*4, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.conv3 = nn.Conv1d(self.seq_len*4, self.seq_len*8, kernel_size=kernel_size, stride=stride, padding=padding)
    
        #self.batch_norm2 = nn.BatchNorm1d(self.seq_len*4)
        #self.batch_norm3 = nn.BatchNorm1d(self.seq_len*8)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0,2,1)
        
        x_conv_list = [F.relu(conv1d(x)) for conv1d in self.conv1_list]

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc(self.dropout(x_fc))
        #x = self.conv1(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)
    
        #x = self.conv2(x)
        #x = self.batch_norm2(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)

        #x = self.conv3(x)
        #x = self.batch_norm3(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)

        #x = torch.flatten(x)

        return logits

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
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False, dropout=0, activation='softmax', approach='qurator', device='cpu'):
        super(GeneratorLSTM, self).__init__()

        self.model_type = 'lstm'
        self.approach = approach # 'qurator' or 'c-rnn-gan'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device


        if self.approach == 'qurator':
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
        elif self.approach == 'c-rnn-gan':
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            if self.bidirectional:
                self.fc2 = nn.Linear(hidden_size*2, output_size)
            else:
                self.fc2 = nn.Linear(hidden_size, output_size)

        # Activation Function !!!

    def forward(self, x, hidden, cell):

        
        if self.approach == 'qurator':

            embedded = self.embedding(x).view(self.batch_size, 1, -1)
            lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

            # not sure; maybe wrong dimensions
            linear_output = self.fc(lstm_output).view(self.batch_size, -1)
        elif self.approach == 'c-rnn-gan':
            pass

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

class GeneratorTransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(GeneratorTransformer, self).__init__()

        self.model_type = 'transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.linear(ninp, ntoken)

        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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
