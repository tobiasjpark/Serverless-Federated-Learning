# Serverless-Federated-Learning
Master's Project to use federated learning to train machine learning models over edge devices and develop a federated learning framework using the serverless computing model on AWS Lambda/Greengrass.

# TODO
Tuesday:
- Get everything set up again with one client
- How to split the data up between clients? Goal: Script that takes number of clients, creates that many sets of data; one set is passeed to each client.
  - Code takes pickled objects or pytorch dataloaders. Script to split up will split up the dataloader and pickle that.
- How does the server specify which clients are participating? Should be simple, instead of putting model in bucket, put file with that data in bucket

Wednesday:
- Modify the code to make it organized, modular, and customizable
  - Customizable, modular algorithm for how server averages models together
  - How does the server know when to continue with a round after hearing back from some clients? Timeout + threshold. Should be customizable/modular.

Thursday:
  - Customizable, modular algorithm for how often clients check for update. Exponential weighted averaging?
  - Customizable, modular algorithm for how the server chooses which clients participate
  
Weekend:
- Refactor remaining code to implement good software design that can be easily used by others
- Documentation

