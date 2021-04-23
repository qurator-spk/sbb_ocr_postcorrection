import torch 
import torch.nn as nn 
import torch.functional as F

class ArgMaxConverter(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(ArgMaxConverter, self).__init__()

        self.model_type = 'linear'

        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size 
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, output_size)
        #self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(0.2) 
        self.dropout_hidden = nn.Dropout(0.5)
   
    def forward(self, x):


        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout_hidden(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.dropout_hidden(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc4(x)

        #x = self.relu(x)

        #x = self.softmax(x)

        x = self.sigmoid(x)

        return x


class ArgMaxConverterCNN(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
#                filter_sizes=[3,4,5],
#                num_filters=[100, 100, 100],
#                num_classes = 2,                
                kernel_size=2, 
                stride=2, 
                padding=1): 
#                lrelu_neg_slope=0.2,
#                dropout_prob=0.5):
        super(ArgMaxConverterCNN, self).__init__()

        self.model_type = 'cnn'

        self.seq_len = 40

        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        #self.lrelu_neg_slope = lrelu_neg_slope

        #self.one_hot_to_emb = nn.Linear(input_size, hidden_size)

        #self.conv1_list = nn.ModuleList([
        #    nn.Conv1d(in_channels=hidden_size,
        #              out_channels=num_filters[i],
        #              kernel_size=filter_sizes[i])
        #    for i in range(len(filter_sizes))
        #])

        #self.fc = nn.Linear(np.sum(num_filters), num_classes)
        #self.dropout = nn.Dropout(p=dropout_prob)

        # Encoder
        self.conv1 = nn.Conv2d(40, 16 , kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.conv3 = nn.Conv1d(self.seq_len*4, self.seq_len*8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(2,2)

        #self.batch_norm2 = nn.BatchNorm1d(self.seq_len*4)
        #self.batch_norm3 = nn.BatchNorm1d(self.seq_len*8)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):

        #x = self.embedding(x)

       # x = self.one_hot_to_emb(x)

        #import pdb; pdb.set_trace()

        #x = x.permute(0,2,1)
        
        #x_conv_list = [F.relu(conv1d(x)) for conv1d in self.conv1_list]

        #x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #    for x_conv in x_conv_list]

        #x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        #logits = self.fc(self.dropout(x_fc))

        #if self.num_classes == 1:
        #    logits = logits.squeeze()
        #x = self.conv1(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)
    
        #x = self.conv2(x)
        #x = self.batch_norm2(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)

        #x = self.conv3(x)
        #x = self.batch_norm3(x)
        #x = F.leaky_relu(x, negative_slope=self.lrelu_neg_slope)

        #x = torch.flatten(x)

        x = x.unsqueeze(3)

        import pdb; pdb.set_trace()

        # Encoding
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Decoding
        x = self.t_conv1(x)
        x = self.relu(x)
        x = self.t_conv2(x)
        #x = F.relu(x)

        x = F.sigmoid(x)

        return x