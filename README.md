# Serverless-Federated-Learning
Master's Project to use federated learning to train machine learning models over edge devices and develop a federated learning framework using the serverless computing model on AWS Lambda/Greengrass.

# TODO
- Verify that the one-client system works with Greengrass
- Look into Nvidia's new tech and see how they pass the models between servers and clients
- How to split the data up between clients? Goal: Script that takes number of clients, creates that many sets of data; one set is passeed to each client.
- How does the server specify which clients are participating? Should be simple, instead of putting model in bucket, put file with that data in bucket
- How does the server know when to continue with a round after hearing back from some clients? Timeout + threshold. Should be customizable/modular.
- Customizable, modular algorithm for how often clients check for update. Exponential weighted averaging?
- Customizable, modular algorithm for how server averages models together
