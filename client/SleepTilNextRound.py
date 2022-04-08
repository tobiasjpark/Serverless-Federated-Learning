import time

# Edit the sleepTilNextRound function to control how long a client waits 
# in between checking for a new round. It is given an argument 
# completed_round which is True if a round just finished and False 
# otherwise. It is called after every time a client finishes checking 
# for a round.

def sleepTilNextRound(completed_round): # if completed_round == True, round just finished. 
    # Can perform simple math/memorization to figure out how long the round was and how much time passed between past two rounds
    time.sleep(1)