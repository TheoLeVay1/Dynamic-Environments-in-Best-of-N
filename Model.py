import mesa
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def compute_consensus(model):
    agent_opinions = [agent.opinion for agent in model.schedule.agents]
    agreed_opinions = [i for i in agent_opinions if i >= 0.9]
    return len(agreed_opinions) / len(agent_opinions)

def compute_average_opinion(model):
    agent_opinions = [agent.opinion for agent in model.schedule.agents]
    return sum(agent_opinions)/len(agent_opinions)

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
        c_x = ((np.prod(pooled_opinions))**w)/(((np.prod(pooled_opinions))**w)+(np.prod(list(1-np.asarray(pooled_opinions)))**w))
        for agent in pooled_agents:
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

class Model(mesa.Model):

    def __init__(self, K, n, w, alpha, epsilon, pooling = True, uniform = False, dynamics = "none"):
        self.num_agents = K
        self.STEP = 0
        self.pooling = pooling
        self.uniform = uniform
        self.dynamics = dynamics
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


    def step(self):
        self.datacollector.collect(self)
        self.STEP += 1
        self.schedule.step()
