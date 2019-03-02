import numpy as np
import random
random.seed(0)
np.random.seed(0)

class Environment:
  def __init__(self, num_arms=10):
    self.num_arms = num_arms
    
    # for each arm, save a random probability for success
    self._probs = [random.random() for _ in range(self.num_arms)]
    self.action_space = range(num_arms)


  def try_arm(self, arm_num):
    
    # either succed or fail randomly based on the arm's probability
    got_reward = random.random() < self._probs[int(arm_num)]

    return 1.0 if got_reward else 0.0