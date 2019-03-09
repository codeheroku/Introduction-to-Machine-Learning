import numpy as np

class RandomAgent():
	def __init__(self,action_space):
		self.action_space = action_space
	
	def choose_action(self,observation):
		##Todo: Returns a random choice of the available actions
		return self.action_space.sample()


class ValueIterAgent():
	def __init__(self,env,gamma):
		self.max_iterations = 1000
		self.gamma = gamma
		self.num_states=env.observation_space.n
		self.num_actions=env.action_space.n
		self.state_prob = env.env.P

		self.values = np.zeros(env.observation_space.n)
		self.policy = np.zeros(env.observation_space.n)

	def value_iteration(self):
	    for i in range(self.max_iterations):
	        prev_v = np.copy(self.values)
	        for state in range(self.num_states):
	            Q_value = []
	            for action in range(self.num_actions):
	                next_states_rewards = []
	                for trans_prob, next_state, reward_prob, _ in self.state_prob[state][action]: 
	                   
	                    next_states_rewards.append((trans_prob * (reward_prob + self.gamma * prev_v[next_state]))) 
	                
	                Q_value.append(sum(next_states_rewards))

	            
	            self.values[state] = max(Q_value)
	            
	    return self.values
  

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
			
		return self.policy[observation]
		
		







    	

