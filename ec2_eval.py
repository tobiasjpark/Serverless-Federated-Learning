import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle
import boto3
import sys
from time import sleep 

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

# Prints out the current test accuracy of net; to be used on server
def test(net, testloader, criterion): # MODIFY TO ALSO PRINT OUT LOSS
    correct = 0
    loss = 0
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
            loss += criterion(net(inputs), labels).item()

    print('Test Accuracy, Loss: ', (100 * correct / total), loss/len(testloader))


# Load the data
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64

# Define test set on server
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Initialize server's Neural Network 
version = -1
last_version = -1
storage_bucket = boto3.resource('s3').Bucket("global-server-model") # bucket where server stores its global model
while True:
    try:
        contents = storage_bucket.objects.all() 

        skip = True 

        for name in contents:
            filename = name.key
            version = int(filename)
            skip = False 

        if skip:
            continue

        if version > last_version:
            last_version = version
            print(version)
            service_client = boto3.client('s3')
            service_client.download_file('global-server-model', str(version), 'tmp_model.nn')
            net_file = open('tmp_model.nn', 'rb')
            net_dict = pickle.load(net_file)
            my_net = Net() 
            my_net.load_state_dict(net_dict)
            criterion = nn.CrossEntropyLoss()

            test(my_net, testloader, criterion)
    except:
        pass

    sleep(1)

