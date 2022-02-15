import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

global_net = Net()

my_net_dict = global_net.state_dict()
pickle.dump(my_net_dict, open('1', 'wb'))
