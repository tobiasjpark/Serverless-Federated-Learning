import random

# This function is used by the server to choose clients to be assigned in a given round
def chooseClients(clients):
    percentage = 1 # percentage of clients to randomly choose for a given round, given as a decimal

    random.shuffle(clients)
    length = len(clients)
    index = length * percentage
    return clients[0:index]