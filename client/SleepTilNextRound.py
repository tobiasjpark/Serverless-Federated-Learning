import time

# This function runs on the client when it wants to wait a certain amount of time before checking for a new round.
# Edit the function to control the amount of time the client should wait or implement a dynamic algorithm that changes the duration of this wait time.
def sleepTilNextRound(completed_round): # if completed_round == True, round just finished. 
    # Can perform simple math/memorization to figure out how long the round was and how much time passed between past two rounds
    time.sleep(30)