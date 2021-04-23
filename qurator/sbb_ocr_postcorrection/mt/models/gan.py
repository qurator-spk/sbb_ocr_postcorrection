import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq2seq import AttnDecoderLSTM, EncoderLSTM

class DiscriminatorCNN(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                filter_sizes=[3,4,5],
                num_filters=[100, 100, 100],
                num_classes = 2,                
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
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.lrelu_neg_slope = lrelu_neg_slope

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.one_hot_to_emb = nn.Linear(input_size, hidden_size)

        self.conv1_list = nn.ModuleList([
            nn.Conv1d(in_channels=input_size,#in_channels=hidden_size,
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

        #x = self.embedding(x)

        #x = self.one_hot_to_emb(x)

        #import pdb; pdb.set_trace()

        x = x.permute(0,2,1)
        
        x_conv_list = [F.relu(conv1d(x)) for conv1d in self.conv1_list]

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc(self.dropout(x_fc))

        if self.num_classes == 1:
            logits = logits.squeeze()
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

class GeneratorSeq2Seq(nn.Module):
    '''
    The Generator Seq2Seq base model.

    It integrates encoder and decoder models for machine translation.

    Inspired by: https://bastings.github.io/annotated_encoder_decoder/
    '''
    def __init__(self, input_size, hidden_size, output_size, batch_size, 
            seq_length, rnn_type, n_layers, bidirectional, dropout, 
            activation, device):
        super(GeneratorSeq2Seq, self).__init__()

        #encoder, decoder, src_embed, trg_embed, rnn_type='LSTM'
        #self.encoder = encoder 
        #self.decoder = decoder 
        #self.src_embed = src_embed 
        #self.trg_embed = trg_embed
        #self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.activation = activation
        self.device = device

        if rnn_type == 'lstm':
            self.encoder = EncoderLSTM(input_size, hidden_size, batch_size, n_layers, dropout, device).to(device)
            self.decoder = AttnDecoderLSTM(hidden_size, output_size, batch_size, seq_length, n_layers, dropout, device).to(device)
        elif rnn_type == 'gru':
            pass 

    def forward(self, input_tensor, target_tensor, input_length, target_length,
        use_teacher_forcing):
        
        #################
        #               #
        # Encoding Step #
        #               #
        #################

        encoder_hidden = self.encoder.init_hidden_state()
        encoder_cell = self.encoder.init_cell_state()

        encoder_outputs = torch.zeros(self.batch_size,
                                target_length,
                                self.hidden_size,
                                device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = self.encoder(
                                                            input_tensor[ei], 
                                                            encoder_hidden, 
                                                            encoder_cell)

            for bi in range(self.batch_size):
                encoder_outputs[bi, ei] = encoder_output[bi, 0]

        #################
        #               #
        # Decoding Step #
        #               #
        #################

        #loss = 0

        decoder_input = input_tensor[0].clone().detach()
        decoder_outputs = torch.zeros(target_length,
                            self.batch_size,
                            self.output_size) # set to 1 for development

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell, \
                    decoder_attention = self.decoder(decoder_input,
                                            decoder_hidden, 
                                            decoder_cell,
                                            encoder_outputs)
                
        #        loss += criterion(decoder_output, target_tensor[di])

                # teacher forcing
                decoder_input = target_tensor[di]
                
                decoder_outputs[di] = decoder_output
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell, \
                    decoder_attention = self.decoder(decoder_input,
                                            decoder_hidden,
                                            decoder_cell,
                                            encoder_outputs)


        #        loss += criterion(decoder_output, target_tensor[di])

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                decoder_outputs[di] = decoder_output
        
        decoder_outputs = decoder_outputs.transpose(0,1)
    
        #decoded_generated_tensor_g = torch.zeros(200, 40, requires_grad=True)

        #for i in range(decoded_generated_tensor_g.shape[0]):
        #    topv, topi = decoder_outputs[i].data.topk(1)

        #    decoded_generated_tensor_g[i, :] = topi.squeeze() 

        return decoder_outputs#decoded_generated_tensor_g#decoder_outputs#, loss

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

class CERLoss(nn.Module):
    def __init__(self):
        super(CERLoss).__init__()

    def forward(self, generated_seq, target_seq):
        loss = None 

        return loss 

class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, output_size)
    
    def forward(self, x):
        return self.embedding(x)


def real_loss(d_logits, criterion, smooth=False, device='cpu'):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape#[0]

    if smooth:
        labels = torch.ones(size, device=device)*0.9
    else:
        labels = torch.ones(size, device=device) # dtype=torch.long

    loss = criterion(d_logits, labels)
    return loss

def fake_loss(d_logits, criterion, device='cpu'):
    '''
    Taken from: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist
    '''
    size = d_logits.shape#[0]

    labels = torch.zeros(size, device=device)#, dtype=torch.long)

    loss = criterion(d_logits, labels)
    return loss


