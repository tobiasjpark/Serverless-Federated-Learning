import random

# Edit the chooseClients function in this file to specify a custom 
# algorithm for how clients should be chosen at the beginning of each 
# global round. 
# 
# The function is called at the beginning of each global 
# round; it is given as an argument a list `clients` of the names of all 
# clients who are available for participation and must return a list of 
# the names of clients chosen to participate in the next global round.

def chooseClients(clients):
    percentage = 1 # percentage of clients to randomly choose for a given round, given as a decimal

    random.shuffle(clients)
    length = len(clients)
    index = length * percentage
    return clients[0:index]