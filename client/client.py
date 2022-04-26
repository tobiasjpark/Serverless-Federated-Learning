b = 0.8 # client selection algorithm - exponentially weighted average parameter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import boto3
import pickle
from Net import Net
from Train import train
from LoadData import loadData
from SleepTilNextRound import sleepTilNextRound
from CreatePickle import createPickle
import sys, os
from time import time, mktime
from datetime import timezone

class Client:
    def __init__(self, id, epoch):
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        self.CLIENT_ID = id
        self.service_client = boto3.client('s3')

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('clients')
        table.put_item(Item={'device_id': id, 'average': '-1'})

        self.client_data = loadData(str(id))

        # Check every 30 seconds for new model from server with higher version number than what we have (if we have anything)
        # Wait until new model is found
        while True:
            print("Checking for commands from server...")
            completed_round = False
            version = -1
            try:
                self.service_client.download_file('client-assignments', self.CLIENT_ID, '/tmp/' + self.CLIENT_ID + 'tmp.txt')
                version = int(open('/tmp/' + self.CLIENT_ID + 'tmp.txt', 'r').read()) + 1
            except:
                continue

            if version == -1:
                continue

            print("Participating in new round...")
            print("Creating new Version " + str(version))

            version_file = None 
            try:
                version_file = open(self.CLIENT_ID + 'version.txt', 'r')
            except:
                # file does not exist; just say 
                version_file = open(self.CLIENT_ID +'version.txt', 'w')
                version_file.write("0")
                version_file.close()
                version_file = open(self.CLIENT_ID +'version.txt', 'r')
            current_version = int(version_file.read())

            if current_version >= version:
                print("Already processed this version")
                sleepTilNextRound(completed_round)
                continue # no update, we already processed this

            timestamps = {}
            timestamps["T1"] = time()

            print("Downloading current model version from server")
            self.service_client.download_file('global-server-model', str(version-1), self.CLIENT_ID + 'tmp_model.nn')
            timestamps["T2"] = time()
            timestamps["T3 Download Time"] = timestamps["T2"] - timestamps["T1"] 
            net_file = open(self.CLIENT_ID + 'tmp_model.nn', 'rb')
            net_dict = pickle.load(net_file)
            my_net = Net() 
            my_net.load_state_dict(net_dict)

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            my_net.to(device)
            cudnn.benchmark = True
            
            print("Training locally...")
            # Perform update step locally
            my_net = train(my_net, self.client_data, epoch) 

            timestamps["T4"] = time()
            timestamps["T5 Compute Time"] = timestamps["T4"] - timestamps["T2"] 

            print("Training finished, uploading new model to server")

            # Upload to server
            my_net_dict = createPickle(my_net, self.client_data)
            pickle.dump(my_net_dict, open(self.CLIENT_ID + 'tmp_net.nn', 'wb'))
            upload_filename = str(str(version) + "-" + str(self.CLIENT_ID))
            self.service_client.upload_file(self.CLIENT_ID + 'tmp_net.nn', 'client-weights', upload_filename)

            version_file = open(self.CLIENT_ID + 'version.txt', 'w')
            version_file.write(str(version))
            version_file.close()
            completed_round = True
            print("Done")

            # get S3 upload timestamp
            response = self.service_client.head_object(Bucket="client-weights", Key=upload_filename)
            t = response["LastModified"]
            t = t.replace(tzinfo=timezone.utc).astimezone(tz=None) # convert from UTC to local time so the timestamps match up for the Upload Time calculation
            timestamps["T6"] = mktime(t.timetuple()) + t.microsecond / 1E6
            timestamps["Upload Time"] = timestamps["T6"] - timestamps["T4"]
            total = timestamps["T3 Download Time"] + timestamps["T5 Compute Time"] + timestamps["Upload Time"] 

            dyn_table = boto3.client('dynamodb')
            for timestamp in timestamps:
                if timestamp in ("T1", "T2", "T4", "T6"):
                    continue
                dyn_table.update_item(TableName='timestamps', Key={'Version': {'N': str(version)}}, AttributeUpdates={self.CLIENT_ID + '-' + timestamp: {'Value': {'N': str(timestamps[timestamp])}}})

            average_exists = dyn_table.get_item(TableName='clients', Key={'device_id': {'S': str(self.CLIENT_ID)}})['Item']['average']['N']
            if average_exists == -1:
                dyn_table.update_item(TableName='clients', Key={'device_id': {'S': str(self.CLIENT_ID)}}, AttributeUpdates={"average": {'Value': {'N': str(total)}}})
            else:
                new_avg = b * float(average_exists) + (1-b) * total
                dyn_table.update_item(TableName='clients', Key={'device_id': {'S': str(self.CLIENT_ID)}}, AttributeUpdates={"average": {'Value': {'N': str(new_avg)}}})

            sleepTilNextRound(completed_round)

if __name__ == '__main__':
    id = sys.argv[1]
    epoch = sys.argv[2]
    Client(id, int(epoch))