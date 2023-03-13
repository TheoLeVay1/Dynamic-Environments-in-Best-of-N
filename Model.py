import mesa
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from plotting_functions import *

'''
Definitions for model-wide parameters that is called later in self.datacollector model_reporters
'''

def compute_majority(self):
    
    agent_opinions = [agent.opinion for agent in self.model.schedule.agents]
    agreed_opinions = [i for i in agent_opinions if i >= 0.9]

    # If loop for the init call of this function, prevents divide by 0 error
    if len(agent_opinions) != 0:
        return len(agreed_opinions) / len(agent_opinions)
    
    else:
        return 0

def compute_average_opinion(model):
    
    agent_opinions = [agent.opinion for agent in model.schedule.agents]
    return sum(agent_opinions)/len(agent_opinions)


'''
Standard agents intialised with:

unique_id: the unique_id for each agent
weight: w parameter to be used in SProdOp equation
opinion: initial opinion between 0 and 1, representing option 0 and option 1
alpha: the 'trust' parameter found in Bayesian updating, i,e, trust in the option being correct
epsilon: the probability of agents being subject to evidence

functions

pool_agents: returns (n = pool_size) pool of agents
SProdOp: SProdOp function as per literature
bayesian_update: Bayesian updating function as per literature
switched_bayesian: Version of Bayesian updating where the evidence switched behaviour after a certain value (currently manually inputted)

'''


class Agent(mesa.Agent):
    
        
    def __init__(self, unique_id, model, w, alpha, epsilon):
        
        super().__init__(unique_id, model)
        # Initialise each agent with an opinion on whether H1 is true. Randomly distributed between 0 and 1.
        
        if self.model.uniform == True:
            self.opinion = 0.5
            
        else:
            self.opinion = random.uniform(0,1)
            
        self.majority = 0
        # Weight is the same for each agent, since we are planning to use special case in SProdOp
        self.weight = w
        self.epsilon = epsilon
        self.TIME = self.model.TIME
        self.alpha = alpha
        
        # Initialising the real time consesus reading 
        
        self.realtime_consensus = compute_majority(self)
    
    def pool_agents(self):
        
        # Return array of [agent and (n) neighbouring agents]
        
        pooled_agents = [self]
        # Random approach - randomly choose n agents, assumes 'Well-mixed' model.
        while len(pooled_agents) < self.model.pool_size:
            other_agent = self.random.choice(self.model.schedule.agents)
            
            if other_agent not in pooled_agents:
                pooled_agents.append(other_agent)
                
        return pooled_agents  
    
    
    
    def SProdOp(self, pooled_agents):
        
        # SProdOp from combining opinion pooling paper for w = const
        
        pooled_opinions = []
        
        for agent in pooled_agents:
            pooled_opinions.append(agent.opinion)
            
        w = self.weight
        
        c_x = ((np.prod(pooled_opinions))**w)  /  ((np.prod(pooled_opinions)**w)  + 
                                                np.prod(list(1-np.asarray(pooled_opinions)))**w )
        
        if math.isnan(c_x) != True:
        
            for agent in pooled_agents:

                # We do not want to change the opinions of stubborn agents, however we want them 
                # to change the opinions of others

                if agent.stubborn != True:
                    agent.opinion = c_x

                
                
    def bayesian_update(self):
        
        x = self.opinion
        
        if self.model.dynamics == "visit_dynamic" or self.model.dynamics == "time_dynamic":
            
            # We reach this function with a probability of epsilon. So now we just need to use option qualities
            # to decide which hypothesis' evidence will be shown.
            # we have q_1 = model.option1_quality; hence q_0 = 1 - model.option1_quality
            
            x = random.uniform(0,1)
            
            if x < self.model.option1_quality:
                delta = 1 - self.alpha
                
            else: 
                delta = self.alpha
                
            if self.model.dynamics == "visit_dynamic":
                if self.TIME > 50:
                    if self.model.option1_quality > 0:
                        self.model.option1_quality -= 0.01
                
        else: 
            
            delta = 1 - self.alpha

        # Bayesian update according to Definition 3.1 from combining opinion pooling paper
        self.opinion = ( delta*x ) / ( (delta*x) + ((1-delta)*(1-x)) )
        
        
        
    def switched_bayesian(self):
                
        if self.model.TIME < 100:
            delta = 1 - self.model.alpha
            
        if self.model.TIME >= 100:
            delta = self.model.alpha  
            self.model.option1_quality = 0

            
        x = self.opinion
        self.opinion = ( delta*x ) / ( (delta*x) + ((1-delta)*(1-x)) )                
            
            
    def step(self):
        # Simulate agents randomly coming comparing the two options
        x = random.uniform(0,1)        
        
        if self.stubborn != True:        
            
            if x < self.epsilon:
                
                if self.model.dynamics == "switching":
                    self.switched_bayesian()
                    
                else:
                    self.bayesian_update()

        # Pooling only occurs once all the agents have 'moved' simultaneously and had a chance of finding evidence, hence [-1]

        if self.model.pooling == True:
            
            if self == self.model.schedule.agents[-1]:
                
                pooled_agents = self.pool_agents()
                self.SProdOp(pooled_agents)
                
                
        # Updating the time of the whole model running
        
        self.realtime_consensus = compute_majority(self)
        self.TIME += 1
        
        self.model.option0_quality = 1 - self.model.option1_quality
                
#         self.model.TIME += 1
                                         
'''
Model class:

K: total number of class agents to call
TIME: time to be updated for every step change in the agent class, rather than at the model level

dynamics:

"none"
"switching": dynamic switching at time (=200 TIME)
"time_dynamic": option quality changes as a function of time 
"visit_dynamic": option quality changes as a function of agent visits

'''                    
                    
class Model(mesa.Model):
    
    def __init__(self, K, n, w, alpha, epsilon, pooling = False, 
                 uniform = False, dynamics = "none", measures = "none",
                s_proportion = 0, TIME = 0):
        
        self.num_agents = K
        self.TIME = TIME
        self.pooling = pooling
        self.uniform = uniform
        self.alpha = alpha
        self.dynamics = dynamics
        self.measures = measures
        self.s_proportion = s_proportion
        # Shuffle the agents so that they are all activated once per step, and this order is shuffled at each step.
        # This is representative of the 'well mixed' scenario
        self.schedule = mesa.time.RandomActivation(self)
        self.pool_size = n
        self.running = True
        
        
        # Introducing option quality
        
        self.option0_quality = 0
        self.option1_quality = 1
    

        for i in range(self.num_agents):
            a = Agent(i, self, w, alpha, epsilon)
            self.schedule.add(a)
#             self.TIME += 1

        self.datacollector = mesa.DataCollector(
            model_reporters = {"Average_opinion" : compute_average_opinion, "Option 0 quality" : "option0_quality",
                              "Option 1 quality" : "option1_quality"},
            
            
            agent_reporters = {"Opinion" : "opinion", "Majority" : compute_majority, "trueconsensus" : "realtime_consensus"} )
        
        if measures == "stubborn":
        
            # Let's say we want 10% stubborn agents in either direction
            n_stubborn = int(self.num_agents * s_proportion)
            # Then we can find the pool of stubborn agents
            stubborn_pos_pool = self.schedule.agents[0:int(n_stubborn/2)]
            stubborn_neg_pool = self.schedule.agents[int(n_stubborn/2):n_stubborn]

            for agent in self.schedule.agents:
                agent.stubborn = False

                if agent in stubborn_pos_pool:
                    agent.stubborn = True
                    agent.opinion = 0.9

                if agent in stubborn_neg_pool:
                    agent.stubborn = True
                    agent.opinion = 0.1
                    
        else: 
            
            for agent in self.schedule.agents:
                agent.stubborn = False
        

    def step(self):
        
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Decreasing H_1's option quality by 0.1 for every full permutation of the agents (model step)
        if self.TIME > 50:
            if self.dynamics == "time_dynamic":
                if self.option1_quality > 0:
                    self.option1_quality -= 0.01

        self.TIME += 1