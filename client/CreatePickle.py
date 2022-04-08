# Edit the createPickle function in this file to prepare 
# the pyTorch neural network to be pickled as a file. 

# It is given an argument my_net (the pyTorch network object) 
# and a reference to the pyTorch DataLoader object created in 
# LoadData.py (see docs). 
# 
# The DataLoader object is included in 
# case the user wants to use data like the number of points in 
# the dataset and include that information in the pickled object 
# to be used later by the weight averager for a custom averaging 
# algorithm. 
# 
# Whatever object is returned by the createPickle function 
# is the object that is serialized and pickled.

def createPickle(my_net, client_data):
    dictionary = {}
    dictionary["net"] = my_net.state_dict()
    dictionary["size"] = len(client_data.dataset)
    return dictionary