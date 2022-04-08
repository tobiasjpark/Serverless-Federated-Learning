import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import sys

# Load the data
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)


indices = torch.randperm(len(trainset))
data_per = int(len(trainset)/4)

for i in range(4):
    subset = None 

    subset = torch.utils.data.Subset(trainset, indices[data_per*i:data_per*(i+1)])

    file = None 
    file = open('split_dataset_' + str(i+iid*4+1) + '.pkl', 'wb')
    pickle.dump(subset, file)
    file.close()