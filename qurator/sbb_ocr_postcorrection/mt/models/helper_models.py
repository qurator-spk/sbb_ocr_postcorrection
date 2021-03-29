import torch 
import torch.nn as nn 

class OneHotConverter(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(OneHotConverter, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size 
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x