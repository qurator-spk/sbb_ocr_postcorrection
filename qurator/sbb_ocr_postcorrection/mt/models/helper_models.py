import torch 
import torch.nn as nn 

class ArgMaxConverter(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(ArgMaxConverter, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size 
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=2)
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

        x = self.softmax(x)

        return x
