lr = 0.0001
momentum=0.9
ROUNDS_IN_EPOCH = 4


import torch.nn as nn
import torch 
import torch.optim as optim

# Function for training for one epoch on a client
def train(net, trainloader):
    for rounds in range(0, ROUNDS_IN_EPOCH):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print('Batch %5d - loss: %.3f' %
                    (i + 1, running_loss / 100))
                running_loss = 0.0
    return net
