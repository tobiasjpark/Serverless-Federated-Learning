# This function runs on the averager. It compares bucket_set (set of clients who have responded) and 
# assigned_set (set of clients who were assigned in this round) 
# and return True if enough clients have responded to perform the averaging
def checkThreshold(bucket_set, assigned_set):
    return bucket_set == assigned_set