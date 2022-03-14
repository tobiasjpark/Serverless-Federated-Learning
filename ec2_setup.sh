#!/bin/bash

sudo apt update
sudo apt install -y python3-pip
pip3 install torch boto3 torchvision
cd  
mkdir .aws
cd .aws
nano credentials
cd ../Serverless-Federated-Learning/client
