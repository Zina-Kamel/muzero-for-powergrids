"""
Stores the history of actions taken by the agent.
"""

class ActionRecord(object):
    def __init__(self, history, action_dim):      
        self.history = list(history)
        self.action_dim = action_dim
    
    def clone(self):
        return ActionRecord(self.history, self.action_dim)
    
    def add_action(self, action):
        self.history.append(action)
    
    def last_action(self):
        return self.history[-1]