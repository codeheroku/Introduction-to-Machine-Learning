import numpy as np

class RandomAgent():
	def __init__(self,action_space):
		self.action_space = action_space
	
	def choose_action(self,observation):
		##Todo: Return a random choice of the available actions
		return ""


class ValueIterAgent():
	def __init__(self,env,gamma):
		self.max_iterations = 1000
		self.gamma = gamma
		self.num_states = env.observation_space.n
		self.num_actions = env.action_space.n
		self.state_prob = env.env.P

		self.values = np.zeros(env.observation_space.n)
		self.policy = np.zeros(env.observation_space.n)

	##Helper Function	
	def extract_policy(self):
	   
	    for s in range(self.num_states):
	        q_sa = np.zeros(self.num_actions)
	        for a in range(self.num_actions):
	            for next_sr in self.state_prob[s][a]:
	                # next_sr is a tuple of (probability, next state, reward, done)
	                p, s_, r, _ = next_sr
	                q_sa[a] += (p * (r + self.gamma * self.values[s_]))
	        self.policy[s] = np.argmax(q_sa)


	def choose_action(self,observation):
		##Todo: Return action based on calculated policy
		return "" 
		
		







    	

