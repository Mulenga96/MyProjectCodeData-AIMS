# -*- coding: utf-8 -*-

import numpy as np
import gym
import ushiriki_policy_engine_library
import  pandas as pd


                                                                 
iterations = 0
environment = gym.make("ChallengePolicy-v0", 
                     userID="61122946-1832-11ea-8d71-362b9e155667",
                     baseuri="https://reward-service.9xxws07e6gx.us-south.codeengine.appdomain.cloud")
environment.reset()


def MalariaCost(x):
    global iterations
    global environment
    if iterations == 0:
        print('Creating environment')
        environment = gym.make("ChallengePolicy-v0", 
                     userID="61122946-1832-11ea-8d71-362b9e155667",
                     baseuri="https://reward-service.9xxws07e6gx.us-south.codeengine.appdomain.cloud")
        environment.reset()

    individual_policies = []
    for individual in range(len(x['IndPosition0'])):
        policy = {}
        j = 0
        for i in range(1, 6):
            itn = x['IndPosition' + str(j)]['Individual' + str(individual)]
            irs = x['IndPosition' + str(j+1)]['Individual' + str(individual)]
            j += 2
            policy[i] = (itn, irs)
        individual_policies.append(policy)

    individual_payoffs = {}
    individuals = []
    for i, policy in enumerate(individual_policies):

        print(f'{2000-iterations} left!')
        iterations += 5

        if iterations >= 2000:
            iterations = 0
            print(f'Iterations reset to {iterations}')

        individual_payoff = environment.step(policy)[1][-1]
        individual = 'Individual' + str(i)
        individuals.append(individual)
        individual_payoffs[individual] = individual_payoff
    
    return pd.Series(data=individual_payoffs, index=individuals)