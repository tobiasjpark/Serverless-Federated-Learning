import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import boto3
from time import sleep
from boto.s3.connection import S3Connection, Bucket, Key
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

CLIENT_ID = 1

service_client = boto3.client('s3')

conn = S3Connection()
src_bucket = Bucket(conn, "client-assignments") # bucket where clients pull model from
dest_bucket = Bucket(conn, "client-weight") # bucket where clients upload their models/weights to

# Load MNIST data to use as our training data
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
indices = torch.randperm(len(trainset))
data_per = int(len(trainset))

subset = torch.utils.data.Subset(trainset, indices[0:data_per])
client_data = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                        shuffle=True, num_workers=2) # ask Tim what num_workers is

# Check every 30 seconds for new model from server with higher version number than what we have (if we have anything)
# Wait until new model is found
while True:
    print("Checking for commands from server...")
    contents = src_bucket.list() 
    for name in contents:
        print("Participating in new round...")
        filename = name.key
        version = int(filename) + 1 # version number we are trying to create in this round
        print("Creating new Version " + str(version))

        version_file = open('version.txt', 'r')
        current_version = int(version_file.read())

        if current_version >= version:
            print("Already processed this version")
            break # no update, we already processed this
        
        print("Downloading current model version from server")
        service_client.download_file('client-assignments', filename, 'tmp_model.nn')
        net_file = open('tmp_model.nn', 'rb')
        net_dict = pickle.load(net_file)
        my_net = Net() 
        my_net.load_state_dict(net_dict)

        criterion = nn.CrossEntropyLoss()
        lr = 0.0001

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        my_net.to(device)
        cudnn.benchmark = True

        ROUNDS_IN_EPOCH = 1
        print("Training locally...")
        # Perform update step locally
        for round in range(ROUNDS_IN_EPOCH):
            my_net = train(my_net, client_data) 

        print("Training finished, uploading new model to server")
        # Upload to server
        my_net_dict = my_net.state_dict()
        pickle.dump(my_net_dict, open('tmp_net.nn', 'wb'))
        service_client.upload_file('tmp_net.nn', 'client-weights', str(str(version) + "-" + str(CLIENT_ID)))

        version_file = open('version.txt', 'w')
        version_file.write(str(version))
        version_file.close()

        print("Done")

    sleep(30)
