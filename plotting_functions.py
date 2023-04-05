import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_average_opinion(results_df, iterations, max_steps, figsize):
    '''
    Run the simulation using params e.g.,

    params = {"K": 50, "n": 5, "w": 0.5, "alpha": 0.2, "epsilon" : 0.02,
              "pooling" : False, "uniform" : False, "dynamics" : "visit_dynamic",
              "measures" : "none", "s_proportion" : 0.2}

    iterations = 4
    max_steps = 100

    results = mesa.batch_run(Model,parameters=params,number_processes=1,
                             data_collection_period=1,display_progress=True,
        iterations = iterations, max_steps=max_steps)

    results_df = pd.DataFrame(results)

    Then call plot_conensus(results_df, iterations, max_steps)

    '''

    data = []

    for it in range(iterations):
        results_it = results_df[results_df.iteration == it]
        data.append(results_it.Average_opinion)

    # Convert from list to np array
    data = np.array(data)

    # For standard deviation plot, work out standard deviation at each step
    std_arr = []
    for row in range(data.shape[1]):
        row_data = data[:, row]
        std = np.std(row_data)
        std_arr.append(abs(std))


    plt.figure(figsize = figsize)
    plt.plot(results_it.Step, np.mean(data , axis = 0),
    color = 'black', label = 'Mean average opinion')
    
    # Plotting the range of the results by standard deviations
    plt.fill_between(results_it.Step, np.mean(data, axis = 0) + std_arr,
     np.mean(data, axis = 0) - std_arr, color = 'grey', label = 'Standard deviation')
    
    # Now we want to plot the actual, stepping function of option quality
    plt.plot(results_it.Step, results_df[results_df.iteration == 0]["Option 1 quality"],
    color = 'blue', label = 'Stepping function')


    plt.legend(loc = 5)
    plt.xlim(0, max_steps)
    plt.ylim(-0.1, 1.1)
    

def return_consensus_time(results_df, params, iterations, max_steps, variable_parameter1 = "none",
                          variable_parameter2 = "none", reconvergence = False):
    
    '''    

    return_consensus_time:
    
    results_df: The results dataframe created by running the model with set parameters, params
    params: the model parameters
    iterations: the number of iterations that the model was run for
    max_steps: the maximum number of steps that was set when the model was run
    variable_parameters(1/2): up to 2 variable parameters to that the model was run for
    reconvergence: If the case is dynamic, and we want to find the time to the second consensus, set as True

    This function returns an array of shape [len(variable_parameter1), len(variable_parameter2)].
    
    So if there are no variable parameters set, it will return a single value. 
    If there is one variable_parameter1, then it will return a 1 x len(variable_parameter1) array
    If there are two variable params, then it will return shape [len(variable_parameter1), len(variable_parameter2)]
    - This last version can be used to produce heatmap plots

    
    '''
    
    if variable_parameter1 == "none":
        data = []
        
        failCount = 0 
        
        for it in range(iterations):         
            # At the point at which the majority is at the consensus point (0.9), consensus is reached
            # We split the dataframe again by iteration
            
            if reconvergence == False:   
                results_it = results_df[(results_df.iteration == it) & (results_df.Majority >= 0.9)]
                
                if len(results_it) > 0:
                    consensus_time = results_it.Step.values[0]
                    
                else:
                    consensus_time = params['dynamic_point'] ## from start to dynamic point
                    failCount += 1
                
            # Reconvergence -> Dynamic Majority (proportion of opinions 0.1 or below)
            if reconvergence == True:
                results_it = results_df[(results_df.iteration == it) & (results_df.Dynamic_Majority >= 0.9) 
                                        & (results_df.Step > params['dynamic_point']) ]
                
                if len(results_it) > 0:
                                        
                    consensus_time = results_it.Step.values[0] - params["dynamic_point"]
                    
                else:
                    consensus_time = max_steps - params["dynamic_point"] 
                    failCount += 1
                    
            data.append(consensus_time)
            
        time = np.mean(np.array(data), axis = 0)
        std = np.std(np.array(data))
                
        return time, std, failCount
     
        
    # In the case of varying a parameter for sake of comparison, e.g. SProdOp parameter w
        
    else:
        
        data_mat = np.empty(shape = (len(params[variable_parameter1]), len(params[variable_parameter2])) )
        std_mat = np.empty(shape = (len(params[variable_parameter1]), len(params[variable_parameter2])) )
        fail_mat = np.empty(shape = (len(params[variable_parameter1]), len(params[variable_parameter2])) )        

        idx = -1
        
        for p1 in params[variable_parameter1]:
            idx += 1
            consensus_avgs = []
            std_arr = []
            fail_arr = []
            results_p1 = results_df[results_df[variable_parameter1] == p1]

            for p2 in params[variable_parameter2]:                   
                data = [] 
                # Split the dataframe into separate dataframes for each parameter value
                results_p2 = results_p1[results_p1[variable_parameter2] == p2]
                
                # Recording the number of fails; i.e, not converging in time
                failCount = 0
                
                for it in range(iterations):
                                    
                    if reconvergence == False:   
                        results_it = results_p2[(results_p2.iteration == it) & (results_p2.Majority >= 0.9)]
                        
                        if len(results_it) > 0:
                            consensus_time = results_it.Step.values[0]
                        else:
                            failCount += 1
                            consensus_time = params['dynamic_point'] ## from start to dynamic point

                    # Reconvergence -> Dynamic Majority (proportion of opinions 0.1 or below)
                    if reconvergence == True:
                        results_it = results_p2[(results_p2.iteration == it) & (results_p2.Dynamic_Majority >= 0.9)
                                               & (results_df.Step > params['dynamic_point']) ]
                        
                        if len(results_it) > 0:
                            consensus_time = results_it.Step.values[0] - params["dynamic_point"]
                        else:
                            consensus_time = max_steps - params["dynamic_point"] 
                            failCount += 1
                            
                    data.append(consensus_time)      

                std = np.std(np.array(data))
                std_arr.append(std)
                fail_arr.append(failCount)
                consensus_avgs.append(np.mean(np.array(data), axis = 0))               
                                
            std_mat[idx] = std_arr
            data_mat[idx] = consensus_avgs
            fail_mat[idx] = fail_arr
            
        return data_mat, std_mat, fail_mat
    
    
    
    


def plot_gain(results_df1, results_df2, iterations, params, variable_parameter = "none", reconvergence = False):
    
    
    '''
    
    Given two dataframes, that have been simulated using the same parameters, including one, variable parameter,
    for iterations, this function plots the Gain, where gain is the difference in times to consensus.
    
    The two dataframes require one different parameter. This could be pooling = true vs false, measures = none vs stubborn etc.
    
    '''
    
    
    ## Here we call the return consensus time  function for the two cases. For reconvergence times, we set reconvergence to be true
    
    consensus_array1 = return_consensus_time(results_df1, params, iterations, 
                                             variable_parameter = variable_parameter, reconvergence = reconvergence)
    
    consensus_array2 = return_consensus_time(results_df2, params, iterations, 
                                             variable_parameter = variable_parameter, reconvergence = reconvergence)
    
    gain = consensus_array2 - consensus_array1
    
    positive_gain = np.where(gain < 0, 0, 1)    
    negative_gain = np.where(gain > 0, 0, 1)
    
    plt.plot(params[variable_parameter], consensus_array1, color = 'red', label = 'Simulation 1')
    plt.plot(params[variable_parameter], consensus_array2, color = 'green', label = 'Simulation 2')

    plt.fill_between(params[variable_parameter], consensus_array1, consensus_array2,
                     where = positive_gain, 
                     interpolate = True, color = 'red', label = 'Negative Gain')


    # Let's find the params[variable_par..] where the gain is positive
    # So its the locations of positive_gain, represented in params[variable...]
    
    plt.fill_between(params[variable_parameter], consensus_array1, consensus_array2,
                     where = negative_gain, 
                     interpolate = True, color = 'green', label = 'Positive Gain')
        
    plt.ylabel("Time to consensus")
    plt.xlabel(variable_parameter)
    plt.legend(loc = 5)
    
    return gain


        
def plot_stepping_function(results_df, iterations, max_steps, switch_point):
    '''
    
    This function is to produce the stepping functions alone, solely for demonstration purposes
    
    The switch point changes for the different methods, and so can be changed accordingly

    '''

    data = []

    for it in range(iterations):
        results_it = results_df[results_df.iteration == it]
        data.append(results_it.Average_opinion)

    # Convert from list to np array
    data = np.array(data)
    ax = plt.figure()
    
    # Now we want to plot the actual, stepping function of option quality
    
    plt.axvline( x = switch_point, color = '0.8', linestyle = 'dashed', label = "Average time of \n first consensus")

    plt.plot(results_it.Time, results_df[results_df.iteration == 0]["Option 0 quality"],
    color = 'blue', label = 'Option 0 Quality')
    
    plt.plot(results_it.Time, results_df[results_df.iteration == 0]["Option 1 quality"],
    color = 'gray', label = 'Option 1 Quality')
    
    plt.ylabel('Option quality')
    plt.xlabel('Time')
    
    plt.legend(loc = 5, fontsize = 10)
    plt.xlim(0, max_steps)
    plt.ylim(-0.1, 1.1)
