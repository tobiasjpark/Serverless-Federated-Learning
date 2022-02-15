import boto3

# This function runs on the averager server at the end of a round.
# Using a boto3 API calls, it can be used to update the timer that controls how long to wait before initiating the next round
# length is the final length of the round that just completed.
def updateInitiatorTimer(length):
    client = boto3.client('events')

    ### Insert code/algorithm to calculate new time based on length here ###
    new_length = 0
    # client.put_rule(Name="invoke-initiator", ScheduleExpression="rate(" + str(new_length) + " minutes")