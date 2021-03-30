import torch 
import torch.nn as nn 

class ArgMaxConverter(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(ArgMaxConverter, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size 
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.softmax(dim=2)
    
    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)