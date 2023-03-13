# Dynamic-Environments-in-Best-of-N

Final year project modelling the behaviour of multi-agent systems (swarms) with a dynamic Best-of-N decision problem.

To generate results, run the model, imported from model.py with set parameters e.g,

params = {"K": 100, "n": 10, "w": 0.5, "alpha": 0.05, "epsilon" : 0.02,
          "pooling" : True, "uniform" : True, "dynamics" : "time_dynamic",
          "measures" : "stubborn", "s_proportion" : 0}

iterations = 10
max_steps = 200

results = mesa.batch_run(Model,parameters=params,number_processes=1,
                         data_collection_period=1,display_progress=True,
    iterations = iterations, max_steps=max_steps)

results_df = pd.DataFrame(results)


The dataframe can be processed into useful graphs using the plotting functions, whose usage is described in plotting_function.py