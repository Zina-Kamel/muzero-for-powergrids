from muzero_config import MuZeroConfig
import numpy as np

"""
ReplayBuffer is where the grid interaction data (rollouts) are stored. 
"""

class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.buffer = []
    self.window_size = config.window_size
    self.batch_size = config.batch_size

  def save_rollout(self, rollout):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(rollout)
    
  def last_rollout(self):
    return self.buffer[-1]
  
  def sample_batch(self, future_steps, td_steps, action_dim):
    batch_size = min(self.batch_size, len(self.buffer))
    sampled_rollouts = np.random.choice(self.buffer, size=batch_size, replace=False)
    
    batch = []
    for rollout in sampled_rollouts:
        obs_index = self.sample_obs_from_rollout(rollout)
        observation = rollout.get_observation(obs_index)
        action_history = rollout.history[obs_index : obs_index + future_steps]
        target = rollout.make_target(obs_index, future_steps, td_steps, action_dim)
        
        batch.append((observation, action_history, target))
    
    return batch

  def sample_obs_from_rollout(self, rollout) -> int:
    return np.random.choice(range(len(rollout.observations)))
    