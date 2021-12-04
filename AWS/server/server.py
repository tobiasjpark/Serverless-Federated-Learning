import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import boto3
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

# Prints out the current test accuracy of net; to be used on server
def test(net, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch done, server averaged client weights. Test Accuracy: %d %%' % (100 * correct / total))

service_client = boto3.client('s3')

# Load the test data set to check accuracy
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Each client performs update step and uploads results to server
# Collect all of the model files from the clients, unpickle them, and turn them back into NNs with load_state_dict 

# Get version number of current model stored
storage_bucket = boto3.resource('s3').Bucket("global-server-model") # bucket where server stores its global model
contents = storage_bucket.objects.all() 
version = -1
for name in contents:
    filename = name.key
    version = int(filename) + 1 # new version we are trying to make
print("New server version: " + str(version))

client_nets_results = []
dest_bucket = boto3.resource('s3').Bucket("client-weights") 
contents = dest_bucket.objects.all() 
for name in contents:
    filename = name.key
    if int(filename.split('-')[0]) != version:
        print("Old version detected, discarding")
        continue # old update for an old round, ignore
    service_client.download_file('client-weights', filename, 'tmp_model.nn')
    net_file = open('tmp_model.nn', 'rb')
    net_dict = pickle.load(net_file)
    my_net = Net() 
    my_net.load_state_dict(net_dict)
    client_nets_results.append(my_net)

# Server update step
global_net = average_models(client_nets_results)
test(global_net, testloader)

# save new model in storage bucket, clear other buckets
src_bucket = boto3.resource('s3').Bucket("client-assignments")
resource = boto3.resource('s3')
for x in src_bucket.objects.all():
    resource.Object('client-assignments', x.key).delete()
for x in dest_bucket.objects.all():
    resource.Object('client-weights', x.key).delete()
for x in storage_bucket.objects.all():
    resource.Object('global-server-model', x.key).delete()

my_net_dict = global_net.state_dict()
pickle.dump(my_net_dict, open('tmp_net.nn', 'wb'))
service_client.upload_file('tmp_net.nn', 'global-server-model', str(version))
