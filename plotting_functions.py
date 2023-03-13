import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_average_opinion(results_df, iterations, max_steps):
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


    plt.figure()
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
    
    

def return_consensus_time(results_df, params, iterations, variable_parameter = "none", reconvergence = False):
    
    '''    
    Run simulation, as shown in plot_average_opinion description, to return results_df.
    Except this time, we are interested in varying a parameter, so set the variable_parameter e.g, "w".
    
    '''
    
    if variable_parameter == "none":
        
        data = []
        
        for it in range(iterations):         
            # At the point at which the majority is at the consensus point (0.9), consensus is reached
            # We split the dataframe again by iteration
            
            if reconvergence == False:   
                results_it = results_df[(results_df.iteration == it) & (results_df.Majority >= 0.9)]
                
            elif reconvergence == True:
                results_it = results_df[(results_df.iteration == it) & (results_df.Dynamic_Majority >= 0.9)]
                
            if len(results_it) > 0:
                consensus_time = results_it.Step.values[0]
                data.append(consensus_time)
                    
            else:
                # Here, we are saying that the consensus time is 1000 if the consensus is not reached
                consensus_time = 1000
                data.append(consensus_time)
                    
        return np.mean(np.array(data), axis = 0)
     
        
    # In the case of varying a parameter for sake of comparison, e.g. SProdOp parameter w
        
    else:
        
        consensus_avgs = []
        
        for i in params[variable_parameter]:    
            data = [] 
            # Split the dataframe into separate dataframes for each parameter value
            results_i = results_df[results_df[variable_parameter] == i]

            for it in range(iterations):
                # At the point at which the majority is at the consensus point (0.9), consensus is reached
                # We split the dataframe again by iteration
                
                if reconvergence == False:              
                    results_it = results_i[(results_i.iteration == it) & (results_i.Majority >= 0.9)]

                elif reconvergence == True:
                    results_it = results_i[(results_i.iteration == it) & (results_i.Dynamic_Majority >= 0.9)]

                if len(results_it) > 0:
                    consensus_time = results_it.Step.values[0]
                    data.append(consensus_time)

                else:
                # Here, we are saying that the consensus time is 1000 if the consensus is not reached
                    consensus_time = 1000
                    data.append(consensus_time)

            # Using the mean average between iterations
            consensus_avgs.append(np.mean(np.array(data), axis = 0))
            
        return np.asarray(consensus_avgs)


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

    plt.plot(results_it.Step, results_df[results_df.iteration == 0]["Option 0 quality"],
    color = 'blue', label = 'Option 0 Quality')
    
    plt.plot(results_it.Step, results_df[results_df.iteration == 0]["Option 1 quality"],
    color = 'gray', label = 'Option 1 Quality')
    
    plt.ylabel('Option quality')
    plt.xlabel('Time')
    
    plt.legend(loc = 5, fontsize = 10)
    plt.xlim(0, max_steps)
    plt.ylim(-0.1, 1.1)
