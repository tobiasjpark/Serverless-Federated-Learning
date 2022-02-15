import torch

# This function creates and returns a pytorch dataloader. Modify this code to use your own pytorch dataloader.
def loadData():
    import pickle
    file = open('dataset.pkl', 'rb')
    subset = pickle.load(file)
    return torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)