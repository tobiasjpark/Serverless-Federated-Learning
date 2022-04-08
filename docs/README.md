This project provides a framework for using federated learning to train machine learning models over edge devices. It is a federated learning framework that uses the serverless computing model on AWS Lambda/Greengrass. The application is written in Python 3 and uses PyTorch for all machine learning tasks.

# Architecture

The application uses S3 for storage and DynamoDB to keep track of a database of clients and for mutex locks. The rest of the project has four main components: the global model, the initiator, the group of clients, and the averager. 

## Global Model
This is simply a PyTorch machine learning model which is stored in S3. It represents the most up-to-date model. The global model will be updated in each round of federated learning.

## [Initiator](https://github.com/tobiasjpark/Serverless-Federated-Learning/docs/initiator.md)
The initiator is an AWS Lambda function that initiates the beginning of a single round of training. It can be invoked manually through an API or by a timer at set intervals. When invoked, it kick starts a new round by selecting a subset of clients and notifying them to participate in the new round.

## [Clients](https://github.com/tobiasjpark/Serverless-Federated-Learning/docs/clients.md)
The clients are any number of edge devices running our client code in AWS Greengrass. Each one has a dataset to train with stored locally on the device. Each client checks an S3 bucket periodically to detect the start of a new round. When the Initiator begins a new round, the clients see this and check whether they have been selected to participate. If so, they download the current global model, perform updates/training on it using their local dataset, and upload their resulting models to another S3 bucket.

## [Averager](https://github.com/tobiasjpark/Serverless-Federated-Learning/docs/averager.md)
The averager is an AWS Lambda function. After a certain number of time has passed, or after enough clients have responded with their individual models, the averager takes all of the individual models returned by the clients and averages them together into a single model to create the new global model.

# Setting Up
1. Download all the files from this repo into a folder.
2. Create S3 buckets with the following names:
    - client-assignments
    - client-weights
    - global-server-model
3. Deploy the Lambda functions
    - Initiator
      - Go into the initiator folder and package all of its contents into a ZIP file
      - In the AWS Management Console, create an AWS Lambda function named "initiator" and upload the ZIP file you created as the code
      - The Lambda Function must have an Execution Role that has the Permission Policies AmazonS3FullAccess, AmazonDynamoDBFullAccess, and AWSLambda_FullAccess 
      - Go to AWS Eventbridge and create a rule called "invoke-initiator". Under "Define Pattern", choose "Schedule" and enter a time interval. Under "Select Targets", select "Lambda Function" and choose the initiator function. Disable the rule for now.
    - Averager
      - The averager needs the PyTorch library in order to average the PyTorch models together. Because the PyTorch library is very large, the averager cannot be uploaded to AWS Lambda normally. Instead, we must use the following method as described here: https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/ . This essentially packages the function and its pytorch dependencies into a Docker file and uploads it to an Amazon ECR repository where it can be used as code for a Lambda function. This is summarized in the following steps:
        - Prerequisites:
          - The AWS Command Line Interface (AWS CLI) installed and configured to interact with AWS services locally
          - The AWS Serverless Application Model (AWS SAM) CLI installed
          - The Docker CLI
        - Create an Amazon ECR repository in AWS and register the local Docker to it. The repositoryUri is displayed in the output; save it for later.
          - Create an ECR repository: `aws ecr create-repository --repository-name lambda-pytorch-example --image-scanning-configuration scanOnPush=true --region <REGION>`

          - Register docker to ECR: `aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin `

        - In the "weight-averager" folder, run sam build && sam deploy –-guided.
        - For Stack Name, enter weight-averager.
        - Choose the same Region that you created the Amazon ECR repository in.
        - Enter the image repository for the function (enter the earlier saved repositoryUri of the Amazon ECR repository).
        - For **Confirm changes before deploy** and **Allow SAM CLI IAM role creation**, keep the defaults.
        - For **pytorchEndpoint may not have authorization defined, Is this okay?**, select y.
        - Keep the defaults for the remaining prompts.
          - If you get a Docker error while running these commands, you may need to run `sudo chmod 666 /var/run/docker.sock`. Also, it appears that the Docker images/containers created during this process are not cleaned up automatically afterwards; you may want to pay attention to this especially if your system is low on disk space.
        - After the function is uploaded and set up, find it in the AWS Management Console. Under Configuration -> Permissions, give it an Execution Role that has the Permission Policies AmazonS3FullAccess, AmazonDynamoDBFullAccess, and AWSLambda_FullAccess 
        - In the AWS Management Console, in "Function Overview", click "Add Trigger" and add the bucket "client-weights" for object created events.
4. Set up DynamoDB
  - Create a table named "clients" with primary partition key "device_id" (String). This is used to keep track of all registered clients.
  - Create a table named "mutex-locked-clients" with primary partition key "ResourceId" (Number). This is used to implement mutex locking.
5. Set up clients
  - Note: The client code uses the hostname of the client device as its unique ID for the serverless federated learning system. For this reason, make sure that each client has a different hostname. Additionally, the semicolon character `;` should not be present in any hostname as this is a special character.
  - Make sure a relatively up-to-date version of Python 3 is installed on the client device.
  - Follow steps 1-3 to set up Greengrass V2 on each of your client devices: https://docs.aws.amazon.com/greengrass/v2/developerguide/getting-started.html#install-greengrass-v2
  - Greengrass V2 runs the client code in a separate Linux user account. To ensure that the client code has access to certain python libraries, they must be installed as root. 
  - Go to the client folder and run the following command to deploy the client code to your client device:

```
sudo /greengrass/v2/bin/greengrass-cli deployment create \
  --recipeDir ~/greengrasstest/recipes \
  --artifactDir ~/greengrasstest/artifacts \
  --merge "com.example.HelloWorld=1.0.0"
```

  - If you desire, follow steps 5 and 6 in the above guide to create your component in the AWS IoT Greengrass service and more easily deploy it to multiple client devices with Greengrass V2 installed.