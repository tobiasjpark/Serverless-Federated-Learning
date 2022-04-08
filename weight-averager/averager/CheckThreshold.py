# Edit the checkThreshold function in this file to specify the threshold 
# for how many clients must respond before the averager activates. 
# 
# The arguments bucket_set and assigned_set are sets containing the 
# names of clients who have responded by placing their neural nets in 
# the S3 bucket and the names of all clients that were assigned to 
# participate in this round, respectively. 
# 
# The function should return True if the threshold has been met and False otherwise. 
# 
# The function is called each time the averager detects that a new client response 
# has been received.

def checkThreshold(bucket_set, assigned_set):
    return bucket_set == assigned_set