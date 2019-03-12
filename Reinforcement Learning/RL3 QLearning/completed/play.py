import gym
from Agents import QAgent
import numpy as np
from crawler_env import CrawlingRobotEnv



env = CrawlingRobotEnv(render=False)

agent = QAgent(env,gamma=0.9)



all_rewards=0
current_state=env.reset()
total_reward = 0
i = 0
while i < 900000:
	i=i+1
	action = agent.choose_action(current_state)
	next_state,reward,done,info = env.step(action)
	agent.learn(current_state,action,reward,next_state)
	current_state = next_state
	total_reward += reward

	if i % 5000 == 0: # evaluation
	    print "average_reward in last 5000 steps", total_reward / i
	    if (total_reward / i) > 1.3:
	    	break
	    average_reward = 0
	    env.render = True


env = CrawlingRobotEnv(render=True)
current_state=env.reset()
total_reward = 0
agent.eps = 0

i = 0
while True:
	i=i+1
	action = agent.choose_action(current_state)
	next_state,reward,done,info = env.step(action)
	agent.learn(current_state,action,reward,next_state)
	current_state = next_state
	total_reward += reward

	if i % 5000 == 0: # evaluation
	    print "average_reward in last 5000 steps", total_reward / 5000
	    i=0;average_reward = 0
	    env.render = True
   	

   	

