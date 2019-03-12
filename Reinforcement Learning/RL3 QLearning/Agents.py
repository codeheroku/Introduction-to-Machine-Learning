import random
from collections import defaultdict
import numpy as np
class QAgent():
    def __init__(self,env,gamma):
        
        self.gamma = gamma
        self.env = env
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = 0.2
        self.eps = 0.5   

    def choose_action(self,state):
        ##Todo: Epsilon Greed Action Selector
        return ""

        
    def learn(self,cur_state,action,reward,next_state):
        ##Todo: Learn From Your Experience
        return "" 
     
