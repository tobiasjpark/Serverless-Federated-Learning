# Edit this file to describe the pyTorch neural network being used to 
# train. This should match the Net.py file used by the Averager 
# (see Averager section in the docs). This file will be treated as 
# the neural network object throuhout the rest of the code.

import torch.nn as nn
import torch
import torch.nn.functional as F

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