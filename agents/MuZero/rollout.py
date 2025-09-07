from action_record import ActionRecord
from action import Action
import numpy as np

"""
Rollout object is what gets collected every training iteration. It starts with the first observation and then rolls out
"""

class Rollout(object):
    
    def __init__(self, action_dim, discount, env):
        self.env = env
        self.history = []
        self.rewards = []
        self.observations = []
        self.child_visits = []
        self.root_values = []
        self.action_dim = action_dim
        self.discount = discount
        self.done = False
        self.action_space_size = self.action_dim
    
    def take_action(self, action):
        next_obs, reward, terminations, truncations, infos = self.env.step(action)
        if terminations:
            self.done = True
        self.add_observation(next_obs)
        if not self.done:
            self.rewards.append(reward)
        self.history.append(action)
        return next_obs, infos, terminations
 
    def get_observation(self, i):
        return self.observations[i]
    
    def get_action_history(self):
        return ActionRecord(self.history, self.action_space_size)
    
    def total_rewards(self):
        return sum(self.rewards)

    def add_observation(self, observation):
        self.observations.append(observation)
    
    def store_stats(self, root):
        
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[action].visit_count / sum_visits if action in root.children else 0
            for action in action_space
        ])
        self.root_values.append(root.value())
        
        
    # based off MuZero Paper's implementation
    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, action_space_size: int):
        # target = the discounted root value of the search tree num_unroll_steps steps
        # + the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index].item() * self.discount**td_steps
                
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0.0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                targets.append((0, last_reward, []))
        
        return targets
    
