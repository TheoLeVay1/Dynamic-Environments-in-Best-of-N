import mesa
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

'''
Definitions for model-wide parameters that is called later in self.datacollector model_reporters
'''

def compute_consensus(model):
    
    agent_opinions = [agent.opinion for agent in model.schedule.agents]
    agreed_opinions = [i for i in agent_opinions if i >= 0.9]
    return len(agreed_opinions) / len(agent_opinions)

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
        self.consensus = 0
        # Weight is the same for each agent, since we are planning to use special case in SProdOp
        self.weight = w
        self.alpha = alpha
        self.epsilon = epsilon
                
    
    def pool_agents(self):
        # Want to return the self + n neighbouring agents.
        pooled_agents = [self]
        # Random approach - randomly choose n agents, assumes 'Well-mixed' model.
        while len(pooled_agents) < self.model.pool_size:
            other_agent = self.random.choice(self.model.schedule.agents)
            if other_agent not in pooled_agents:
                pooled_agents.append(other_agent)
        return pooled_agents  
    
    
    
    def SProdOp(self, pooled_agents):
        
        # SProdOp from combining opinion pooling paper. Only works in special case when w = const.
        
        pooled_opinions = []
        
        for agent in pooled_agents:
            pooled_opinions.append(agent.opinion)
            
        w = self.weight
        
        c_x = ((np.prod(pooled_opinions))**w)  /  ((np.prod(pooled_opinions)**w)  + 
                                                np.prod(list(1-np.asarray(pooled_opinions)))**w )
        
        if math.isnan(c_x) == True:
            c_x = 1
        
        for agent in pooled_agents:
            
            # We do not want to change the opinions of stubborn agents, however we want them to change the opinions of others
            
            if agent.stubborn == False:
                agent.opinion = c_x

                
                
    def bayesian_update(self):
        
        # Bayesian update according to Definition 3.1 from combining opinion pooling paper
        delta = 1 - self.alpha           
        x = self.opinion
        self.opinion = ( delta*x ) / ( (delta*x) + ((1-delta)*(1-x)) )
        
        
        
    def switched_bayesian(self):
        
        if self.model.STEP < 200:
            delta = 1 - self.alpha
        if self.model.STEP >= 200:
            delta = self.alpha        
        x = self.opinion
        self.opinion = ( delta*x ) / ( (delta*x) + ((1-delta)*(1-x)) )
                
            
            
    def step(self):
        # Simulate agents randomly coming comparing the two options
        x = random.uniform(0,1)        
        if self.stubborn == False:        
            
            if x < self.epsilon:
                
                if self.model.dynamics == "switching":
                    self.switched_bayesian()
                    
                else:
                    self.bayesian_update()

        # I want to change it so that the pooling only occurs after every agent has moved. This would simulate agents
        # moving simulataneously

        if self.model.pooling == True:
            
            if self == self.model.schedule.agents[-1]:
                
                pooled_agents = self.pool_agents()
                self.SProdOp(pooled_agents)
                         
'''
Model class:

K: total number of class agents to call
STEP: step to be updated for every step change in the agent class, rather than at the model level

dynamics:

"none"
"switching": dynamic switching at time (=200 STEPs)
"time_dynamic": option quality changes as a function of time **YET TO BE INCORPORATED** 
"visit_dynamic": option quality changes as a function of agent visits **YET TO BE INCORPORATED**

'''                    
                    
class Model(mesa.Model):
    
    def __init__(self, K, n, w, alpha, epsilon, pooling = False, 
                 uniform = False, dynamics = "none", measures = "none",
                s_proportion = 0):
        
        self.num_agents = K
        self.STEP = 0
        self.pooling = pooling
        self.uniform = uniform
        self.dynamics = dynamics
        self.measures = measures
        self.s_proportion = s_proportion
        # Shuffle the agents so that they are all activated once per step, and this order is shuffled at each step.
        # This is representative of the 'well mixed' scenario
        self.schedule = mesa.time.RandomActivation(self)
        self.pool_size = n
        self.running = True
    

        for i in range(self.num_agents):
            a = Agent(i, self, w, alpha, epsilon)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
            model_reporters = {"Average_opinion" : compute_average_opinion, "Consensus" : compute_consensus},
            agent_reporters = {"Opinion" : "opinion"})
        
        # Let's say we want 10% stubborn agents in either direction
        n_stubborn = int(self.num_agents * s_proportion)
        # Then we can find the pool of stubborn agents
        stubborn_pos_pool = self.schedule.agents[0:int(n_stubborn/2)]
        stubborn_neg_pool = self.schedule.agents[int(n_stubborn/2):n_stubborn]
        
        for agent in self.schedule.agents:
            agent.stubborn = False
            
            if agent in stubborn_pos_pool:
                agent.stubborn = True
                agent.opinion = 1
                
            if agent in stubborn_neg_pool:
                agent.stubborn = True
                agent.opinion = 0
        

    def step(self):
        self.datacollector.collect(self)
        self.STEP += 1
        self.schedule.step()