import boto3

# Edit the updateInitiatorTimer function if you want to dynamically 
# change the amount of time that the Initiator waits before attempting 
# to start a new round. 
# 
# This can be accomplished by making a boto3 call 
# to the EventBridge timer that triggers the initiator: 
# 
# client.put_rule(Name="invoke-initiator", ScheduleExpression="rate(" + str(new_length) + " minutes") 
# 
# The length of the last round that just completed is passed in as an argument.

def updateInitiatorTimer(length):
    client = boto3.client('events')

    ### Insert code/algorithm to calculate new time based on length here ###
    new_length = 0
    # client.put_rule(Name="invoke-initiator", ScheduleExpression="rate(" + str(new_length) + " minutes")