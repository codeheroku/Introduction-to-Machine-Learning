import gym
from agents import RandomAgent
from agents import ValueIterAgent
import numpy as np

env = gym.make('FrozenLake-v0')


gamma = 1
#agent = RandomAgent(env.action_space)

agent = ValueIterAgent(env,gamma)

agent.value_iteration();
agent.extract_policy();

print "Agent Policy: ", agent.policy
all_rewards=[]
for _ in range(1000):
	obs=env.reset()
	total_reward = 0
	while True:
	    #env.render()
	    action = agent.choose_action(obs)
	    obs,reward,done,info = env.step(action)
	    if done:
	    	all_rewards.append(reward)
	    	break

print "Average Reward: ", np.mean(all_rewards)

   	

