import numpy as np
import random
random.seed(0)
np.random.seed(0)

class RandomAgent():  
	def __init__(self,action_space):
		self.action_space = action_space
      
	# Choose a random action
	def choose_action(self):
		##Todo >> Return a random choice of the available actions
	    

class ValueApproxAgent():
	def __init__(self,action_space,epsilon=0.05):
		self.action_space = action_space
		self.epsilon = epsilon
		self.approx_values = [0.0] * len(action_space)
		self.observation_counts = [0] * len(action_space)


	def choose_action(self):
		##Todo: Returns a random choice of the available actions
	    
	      
	
	def learn(self,action,reward):
		##Todo: Learn from action and reward to update self.approx_values
		







    	

