AVERAGING_ALGO = 0 # 0 = unweighted; 1 = weighted; 2 = use the custom function defined below.



# This function is called by the averager to average all the different models into one single model.
# It takes in a list that contains tuples of type (neural network in dictionary form, # of data points that were used to train).
# It returns a single pytorch neural net in dictionary form.
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