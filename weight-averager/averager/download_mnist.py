import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)