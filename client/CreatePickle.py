def createPickle(my_net, client_data):
    dictionary = {}
    dictionary["net"] = my_net.state_dict()
    dictionary["size"] = len(client_data.dataset)
    return dictionary