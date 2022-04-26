import random
import boto3

# Edit the chooseClients function in this file to specify a custom 
# algorithm for how clients should be chosen at the beginning of each 
# global round. 
# 
# The function is called at the beginning of each global 
# round; it is given as an argument a list `clients` of the names of all 
# clients who are available for participation and must return a list of 
# the names of clients chosen to participate in the next global round.

def chooseClients(clients):
    dyn_table = boto3.client('dynamodb')
    total = 20 
    rankings = {} 

    max_time = -1.0

    for client in clients:
        client_time = float(dyn_table.get_item(TableName='clients', Key={'device_id': {'S': str(client)}})['Item']['average']['S'])
        if client_time > max_time:
            max_time = client_time
        rankings[client] = client_time
    
    sorted_rankings = sorted(rankings, key=rankings.get)

    return sorted_rankings[0:20]
    
