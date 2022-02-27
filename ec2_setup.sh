#!/bin/bash

sudo apt update
sudo apt install python3-pip
pip3 install torch boto3 torchvision
mkdir .aws
cd ../aws
nano credentials
cd ../Serverless-Federated-Learning/client
