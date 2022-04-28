import json
import boto3
import pickle
import time
import traceback
from Net import Net
from AverageModels import AVERAGING_ALGO, customAveragingAlgo 
from CheckThreshold import checkThreshold
from UpdateInitiatorTimer import updateInitiatorTimer

# Unweighted averaging: all models are given equal weight.
# model_states: a list of objects returned by CreatePickle.py in the client code
# returns a new neural network as a state dictionary that represents the averaged model
def unweighted(model_states):
    new_model_state_dict = model_states[0]['net'] # initiate new_model_state_dict with the correct keys. The values will be overwritten later

    # perform the averaging
    for key in new_model_state_dict: # for each parameter in the neural network
        new_val = 0
        denominator = len(model_states)  
        for tmp in model_states: # go through each model and perform the averaging
            model = tmp['net'] 
            new_val += model[key]           
        new_val = new_val / denominator
        new_model_state_dict[key] = new_val 
    return new_model_state_dict

# Weighted averaging: all models are given weight proportional to number of data points
# model_states: a list of objects returned by CreatePickle.py in the client code
# returns a new neural network as a state dictionary that represents the averaged model
def weighted(model_states):
    new_model_state_dict = model_states[0]['net'] # initiate new_model_state_dict with the correct keys. The values will be overwritten later
    for key in new_model_state_dict: # for each parameter in the neural network
        new_val = 0
        denominator = 0
        for tmp in model_states: # go through each model and perform the averaging
            model = tmp['net'] 
            size = int(tmp['size'])

            # Weighted averaging: Each model is given a weight proportional to the number of data points used to train
            for x in range(0, size):
                new_val += model[key]
            denominator += size            
        new_val = new_val / denominator
        new_model_state_dict[key] = new_val 
    
    # Return the new model
    return new_model_state_dict

# Factory method design pattern; return the correct averaging algorithm function
def getAveragingAlgo():
    if AVERAGING_ALGO == 0:
        return unweighted
    elif AVERAGING_ALGO == 1:
        return weighted 
    else:
        return customAveragingAlgo

# Function to average a list of models together into one model
# models is a list that contains tuples of type (neural net, # of data points that were used to train)
# returns the new averaged model
def averageModels(models):
    # Modified from https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008
    # Create an empty Net
    new_model = Net()

    # Run an averaging algorithm that takes the list of model states and averages them together
    averagingAlgo = getAveragingAlgo()
    result_dictionary = averagingAlgo(models)

    # Convert the dictionary into an actual neural net and return it
    new_model.load_state_dict(result_dictionary)
    return new_model

# mutex lock activate
def lock():
    my_resource_id = 1
    try:
        # Put item with conditional expression to acquire the lock
        dyn_table = boto3.resource('dynamodb').Table('mutex-lock-table')
        dyn_table.put_item(
            Item={'ResourceId': my_resource_id},
            ConditionExpression="attribute_not_exists(#r)",
            ExpressionAttributeNames={"#r": "ResourceId"})
        # Lock acquired
        return True
    except:
        return False

# mutex lock deactivate
def unlock():
    dyn_table = boto3.client('dynamodb')
    dyn_table.delete_item(TableName='mutex-lock-table', Key={'ResourceId': {'N': "1"}})

# main function for AWS Lambda
def lambda_handler(event, context):
    timestamps = {}
    timestamps["T7"] = time.time()
    service_client = boto3.client('s3')
    # get list of all clients participating
    assignment_bucket = boto3.resource('s3').Bucket("client-assignments") 
    contents = assignment_bucket.objects.all() 
    json_set = set()
    for name in contents:
        filename = name.key
        json_set.add(filename)

    # get list of clients who responded 
    bucket_set = set()
    dest_bucket = boto3.resource('s3').Bucket("client-weights") 
    contents = dest_bucket.objects.all() 
    for name in contents:
        filename = name.key.split(';')[1]
        bucket_set.add(filename)

    # Check whether enough clients have responded to meet the threshold for continuing
    if not checkThreshold(bucket_set, json_set):
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Improper number of clients, terminating. json: " + str(json_set) + "; bucket: " + str(bucket_set),
                }
            ),
        }

    # Try to acquire a mutex lock (implemented using DynamoDB as described here: https://blog.revolve.team/2020/09/08/implement-mutex-with-dynamodb/). 
    # If we cannot get a lock, that means the threshold was already met and the averaging is already being taken care of by a previous lambda invocation.
    # In this case, we should abort.
    if not lock():
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Failed to acquire a mutex lock; this means another invocation of this function is in the process of finishing this round. Aborting...",
                }
            ),
        }
    version = -1
    try:
        # Each client performs update step and uploads results to server
        # Collect all of the model files from the clients, unpickle them, and turn them back into NNs with load_state_dict 

        # Get version number of current model stored
        storage_bucket = boto3.resource('s3').Bucket("global-server-model") # bucket where server stores its global model
        contents = storage_bucket.objects.all() 
        
        for name in contents:
            filename = name.key
            version = int(filename) + 1 # new version we are trying to make
        print("New server version: " + str(version))

        client_nets_results = []
        dest_bucket = boto3.resource('s3').Bucket("client-weights") 
        contents = dest_bucket.objects.all() 
        for name in contents:
            filename = name.key
            if int(filename.split(';')[0]) != version:
                print("Old version detected, discarding")
                continue # old update for an old round, ignore
            service_client.download_file('client-weights', filename, '/tmp/tmp_model.nn')
            net_file = open('/tmp/tmp_model.nn', 'rb')
            net_dict = pickle.load(net_file)
            client_nets_results.append(net_dict) 

        # Server update step
        global_net = averageModels(client_nets_results)

        # calculate when the round started
        length = time.time()

        # save new model in storage bucket, clear other buckets
        src_bucket = boto3.resource('s3').Bucket("client-assignments")
        resource = boto3.resource('s3')
        for x in src_bucket.objects.all():
            resource.Object('client-assignments', x.key).delete()
        for x in dest_bucket.objects.all():
            resource.Object('client-weights', x.key).delete()
        for x in storage_bucket.objects.all():
            resource.Object('global-server-model', x.key).delete()

        my_net_dict = global_net.state_dict()
        pickle.dump(my_net_dict, open('/tmp/tmp_net.nn', 'wb'))
        service_client.upload_file('/tmp/tmp_net.nn', 'global-server-model', str(version))

        # update initiator timer
        updateInitiatorTimer(time.time() - length)

    except Exception:
        print(str(traceback.format_exc()))
        unlock()
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Error: " + str(traceback.format_exc()),
                }
            ),
        }
    
    unlock()
    timestamps["T8"] = time.time()
    timestamps["T9 Lambda Runtime"] = timestamps["T8"] - timestamps["T7"]

    # get timestamp of upload to global server model 
    response = service_client.head_object(Bucket="global-server-model", Key=str(version))
    t = response["LastModified"]
    timestamps["T11"] = time.mktime(t.timetuple()) + t.microsecond / 1E6

    # download timestamp T0 from initiator from the dynamodb table
    dyn_table = boto3.client('dynamodb')
    response = dyn_table.get_item(TableName='timestamps', Key={'Version':{'N':str(version)}})
    t0 = float(response['Item']['INITIATOR-T0']['N'])
    timestamps["T12 Total Round Time"] = timestamps["T11"] - t0

    for timestamp in timestamps:
        dyn_table.update_item(TableName='timestamps', Key={'Version': {'N': str(version)}}, AttributeUpdates={'AVERAGER-' + timestamp: {'Value': {'N': str(timestamps[timestamp])}}})

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Done!",
            }
        ),
    }

lambda_handler(0, 0)