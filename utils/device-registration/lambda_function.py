# Process device registration requests (add core device name to the database)
import boto3, json

def lambda_handler(event, context):
    storage_bucket = boto3.resource('s3').Bucket("device-registration") 
    s3resource = boto3.resource('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('clients')

    contents = storage_bucket.objects.all()
    for name in contents:
        data = {"device_id": name.key}
        table.put_item(Item=data)
        s3resource.Object('device-registration', name.key).delete()
    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }

