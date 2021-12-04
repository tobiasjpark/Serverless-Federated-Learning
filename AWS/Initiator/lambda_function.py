# literally just copy the model file from the server's storage S3 bucket into the bucket that triggers the clients
import boto3
import json 

def lambda_handler(event, context):
    service_client = boto3.client('s3')
    storage_bucket = boto3.resource('s3').Bucket("global-server-model") # bucket where server stores its global model

    contents = storage_bucket.objects.all()
    for name in contents:
        filename = name.key
        service_client.download_file('global-server-model', filename, '/tmp/' + filename)
        service_client.upload_file('/tmp/' + filename, 'client-assignments', filename) 

    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }

