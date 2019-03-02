import numpy as np
import random
from environment import Environment
from agents import RandomAgent
from agents import ValueApproxAgent
import matplotlib.pyplot as plt
# initialize
env = Environment(20)
agent = RandomAgent(env.action_space)


total_reward = 0
for i in range(1000):
	action = agent.choose_action()
  	total_reward += env.try_arm(action)
  
print('total reward from RandomAgent:', total_reward)



all_rewards = []



agent = ValueApproxAgent(env.action_space,epsilon=0.07)
total_reward = 0
for i in range(1000):
	action = agent.choose_action()
  	reward = env.try_arm(action)
  	agent.learn(action,reward)
 	total_reward += reward

	  	
  
print 'total reward from ValueApproxAgent:', total_reward
print agent.approx_values, env._probs


