# literally just copy the model file from the server's storage S3 bucket into the bucket that triggers the clients
import boto3
import json 
from ChooseClients import chooseClients
from time import time

def lambda_handler(event, context):
    # Make sure there is not currently a round in progress (client assignment and weight buckets should be empty)
    src_bucket = boto3.resource('s3').Bucket("client-assignments")
    dest_bucket = boto3.resource('s3').Bucket("client-weights")
    for x in src_bucket.objects.all():
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "There is still a round in progress. Aborting...",
                }
            ),
        }
    for x in dest_bucket.objects.all():
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "There is still a round in progress. Aborting...",
                }
            ),
        }

    dyn_table = boto3.client('dynamodb')

    # Query the dynamodb database and (for now) just get all of the ids
    # store ids in a json list
    # save json list to tmp
    # upload json list to the client-assignments bucket
    service_client = boto3.client('s3')
    storage_bucket = boto3.resource('s3').Bucket("global-server-model") # bucket where server stores its global model
    version = list(storage_bucket.objects.all())[0].key

    dyn_table.update_item(TableName='timestamps', Key={'Version': {'N': str(int(version)+1)}}, AttributeUpdates={'INITIATOR-T0': {'Value': {'N': str(time())}}})

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('clients')
    responses = table.scan()
    clients = []
    for response in responses['Items']:
        item = response['device_id']
        clients.append(item)
    
    results = chooseClients(clients)

    with open('/tmp/placeholder.txt', 'w') as f:
        f.write(version)
    f.close()
    
    for result in results:
        service_client.upload_file('/tmp/placeholder.txt', 'client-assignments', result) 

    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }

lambda_handler(0, 0)