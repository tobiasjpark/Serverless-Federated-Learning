AVERAGING_ALGO = 0 # 0 = unweighted; 1 = weighted; 2 = use the custom function defined below.

# This file controls which algorithm is used to averager the client 
# models together. 
# 
# Set AVERAGING_ALGO to 0 to use unweighted averaging 
# where each model is given equal weight and averaged. 
# 
# Set it to 1 to perform weighted averaging, where each model is given weight with 
# proportion to the number of datasets that was used to train it. 
# 
# Set it to 2 to perform a custom algorithm, then define that custom 
# algorithm in the customAlveragingAlgo function which takes as argument 
# a list of objects that were pickled in CreatePickle.py on the clients 
# (see client CreatePickle.py documentation) and returns a pyTorch 
# neural net state dictionary. 

def customAveragingAlgo(model_states):
    new_model_state_dict = model_states[0]['net']  # initiate new_model_state_dict with the correct keys. The values will be overwritten later
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