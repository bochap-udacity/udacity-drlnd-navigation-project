import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    '''Action (Policy) Model.'''
    
    def __init__(self, state_size, action_size, hidden_layer_lengths):
        '''Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        '''
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_lengths['fc1'])
        self.fc2 = nn.Linear(hidden_layer_lengths['fc1'], hidden_layer_lengths['fc2'])
        self.fc3 = nn.Linear(hidden_layer_lengths['fc2'], action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)