# Edit the train function to control how the client performs local 
# training on their model. 
# 
# The first 3 lines of this file are constants that set 
# the learning rate and momentum. 
# 
# The function takes the neural net 
# object to train on, the DataLoader from LoadData.py (see above), 
# and the number of epochs (number of local rounds per global round) 
# to train over, respectively. 
# 
# It returns the trained neural net object.

lr = 0.0001
momentum=0.9

import torch.nn as nn
import torch 
import torch.optim as optim

def train(net, trainloader, epoch):
    for rounds in range(0, epoch):
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
