import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import sys

NUM_CLIENTS = int(sys.argv[1])

# Load the data
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
indices = torch.randperm(len(trainset))
data_per = int(len(trainset)/NUM_CLIENTS)

for i in range(NUM_CLIENTS):
    subset = None 
    if i % 2 == 0:
        subset = torch.utils.data.Subset(trainset, indices[data_per*i:data_per*(i+1)-150])
    else:
        subset = torch.utils.data.Subset(trainset, indices[data_per*i-150:data_per*(i+1)])
    file = open('split_dataset_' + str(i) + '.pkl', 'wb')
    pickle.dump(subset, file)
    file.close()