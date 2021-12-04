import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

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

# Function to average a list of models together into one model
def average_models(models):
    # Modified from https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008
    new_model = Net()
    new_model_state_dict = models[0].state_dict()
    model_state_dicts = []
    for model in models:
        model_state_dicts.append(model.state_dict())
    
    for key in model_state_dicts[0]:
        new_val = 0
        for model in model_state_dicts:
            new_val += model[key]
        new_val = new_val / len(models)
        new_model_state_dict[key] = new_val 
    new_model.load_state_dict(new_model_state_dict)

    return new_model
    
# Function for training for one epoch on a client
def train(net, trainloader):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for param in net.parameters():
            param.data = param.data - lr * param.grad.data

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('Batch %5d - loss: %.3f' %
                  (i + 1, running_loss / 100))
            running_loss = 0.0

    return net

# Prints out the current test accuracy of net; to be used on server
def test(net, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch done, server averaged client weights. Test Accuracy: %d %%' % (100 * correct / total))

NUM_CLIENTS = 5

# Load the data
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
indices = torch.randperm(len(trainset))
data_per = int(len(trainset)/NUM_CLIENTS)

# Split training data up and give a subset to each client
trainloader_sub = [] # element i = client i's training ata
for i in range(NUM_CLIENTS):
    subset = torch.utils.data.Subset(trainset, indices[data_per*i:data_per*(i+1)])
    sub_trainset = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                          shuffle=True, num_workers=2) # ask Tim what num_workers is
    trainloader_sub.append(sub_trainset)

# Define test set on server
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Initialize server's Neural Network 
global_net = Net()
criterion = nn.CrossEntropyLoss()
lr = 0.0001

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_net.to(device)
cudnn.benchmark = True

"""
Federated algorithms have 4 main components:
    1. A server-to-client broadcast step.
    2. A local client update step.
    3. A client-to-server upload step.
    4. A server update step.
"""

NUM_EPOCHS = 15
ROUNDS_IN_EPOCH = 1

for epoch in range(NUM_EPOCHS):
    print("\nStarting epoch #" + str(epoch))
    # Broadcast neural network to clients
    client_nets_initial = [] # client's neural networks
    client_nets_results = [] # list containing neural networks uploaded to server by clients
    for i in range(NUM_CLIENTS):
        client_nets_initial.append(copy.deepcopy(global_net))

    # Each client performs update step and uploads results to server
    i = -1
    for net in client_nets_initial:
        i += 1
        my_net = net
        for round in range(ROUNDS_IN_EPOCH):
            print("Updating client #" + str(i))
            my_net = train(my_net, trainloader_sub[i]) 
        client_nets_results.append(my_net)

    # Server update step
    global_net = average_models(client_nets_results)
    test(global_net, testloader)

