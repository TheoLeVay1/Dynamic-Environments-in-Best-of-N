import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_consensus(results_df, iterations, max_steps):
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
     np.mean(data, axis = 0) - std_arr, label = 'Standard deviation')
    # Now we want to plot the actual, stepping function of option quality

    plt.plot(results_it.Step, results_df[results_df.iteration == 0]["Option quality"],
    color = 'grey', label = 'Stepping function')


    plt.legend(loc = 5)
    plt.xlim(0, max_steps)
    plt.ylim(0,1.1)
