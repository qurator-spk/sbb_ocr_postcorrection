import math
import torch
import torch.nn as nn


class EncoderTransformer(nn.Module):
    def __init__(self, input_size, encoder_size, n_head, feedforward_size, n_layers, dropout=0):
        '''
        input_size:

        '''
        super(EncoderTransformer, self).__init__()

        # self.model_type = 'Transformer'
        # self.src_mask = None

        # self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = nn.TransformerEncoderLayer(encoder_size, n_head, feedforward_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)


class PositionalEncoding(nn.Module):

    def __init__(self, encoder_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, encoder_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoder_size, 2).float() * (-math.log(10000.0) / encoder_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
