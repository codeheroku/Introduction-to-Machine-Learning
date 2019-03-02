import numpy as np
import random
random.seed(0)
np.random.seed(0)

class RandomAgent():  
	def __init__(self,action_space):
		self.action_space = action_space
      
	# Choose a random action
	def choose_action(self):
	    """Returns a random choice of the available actions"""
	    return np.random.choice(self.action_space)

class ValueApproxAgent():
	def __init__(self,action_space,epsilon=0.05):
		self.action_space = action_space
		self.epsilon = epsilon
		self.approx_values = [0.0] * len(action_space)
		self.observation_counts = [0] * len(action_space)


	def choose_action(self):
	    """Returns a random choice of the available actions""" 
	    if np.random.uniform(0,1) < self.epsilon:
	    	#print "Making a random choice..." 		
	    	return np.random.choice(self.action_space)
	    else:
	    	return np.argmax(self.approx_values)
	      
	
	def learn(self,action,reward):
		self.observation_counts[action]+=1
		current_val = self.approx_values[action]
		step_size = 1.0 / (self.observation_counts[action])
		self.approx_values[action] = current_val + step_size * (reward - current_val)
 







    	

