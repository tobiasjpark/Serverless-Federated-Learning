import torch

# Edit the loadData function in this file to control how the client's dataset is loaded. 
# The id of the client is given as an argument. 
# The function must return a pyTorch DataLoader object that 
# represents its dataset.

def loadData(id):
    import pickle
    file = open('split_dataset_' + id + '.pkl', 'rb')
    subset = pickle.load(file)
    return torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)