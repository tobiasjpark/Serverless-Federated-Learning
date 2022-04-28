import torch, torchvision
import torchvision.transforms as transforms

# Edit the loadData function in this file to control how the client's dataset is loaded. 
# The id of the client is given as an argument. 
# The function must return a pyTorch DataLoader object that 
# represents its dataset.

def loadData(id):
    transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

    return torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)




