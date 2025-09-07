import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
from typing import Dict, List
from action import Action

class NetworkOutput(typing.NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: Dict[Action, float]
    policy_tensor: torch.Tensor
    hidden_state: torch.Tensor

class PowerGridNetwork(nn.Module):
    def __init__(self, config):
        super(PowerGridNetwork, self).__init__()
        
        self.hidden_layer_size = config.hidden_layer_size
        self.action_dim = config.action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.representation_network = nn.Sequential(
            nn.Linear(config.input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, config.hidden_layer_size),
            nn.LayerNorm(config.hidden_layer_size),
            nn.ReLU()
        )

        self.value_network = nn.Sequential(
            nn.Linear(config.hidden_layer_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_network = nn.Sequential(
            nn.LayerNorm(config.hidden_layer_size),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size * 2, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.action_dim),  # total number of discrete actions
            nn.Softmax(dim=-1)
        )

        self.reward_network = nn.Sequential(
            nn.Linear(config.hidden_layer_size + self.action_dim, config.hidden_layer_size),
            nn.LayerNorm(config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size // 2, 1)
        )

        self.dynamics_network = nn.Sequential(
            nn.Linear(config.hidden_layer_size + self.action_dim, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ReLU()
        )

        self.total_training_steps = 0
        self.to(self.device) 

    def get_training_steps(self):
        return self.total_training_steps

    def increment_training_steps(self):
        self.total_training_steps += 1

    def get_weights(self):
        return [param for param in self.parameters()]

    def initial_inference(self, observation):
        observation = observation.to(self.device)
        hidden_state = self.representation_network(observation)
        value = self.value_network(hidden_state)
        policy = self.policy_network(hidden_state)
        reward = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return NetworkOutput(
            value,
            reward,
            {Action(a): policy[a].item() for a in range(policy.shape[0])},
            policy,
            hidden_state
        )

    def recurrent_inference(self, parent_hidden_state, action):
        parent_hidden_state = parent_hidden_state.to(self.device)
        action_one_hot = torch.eye(self.action_dim, device=self.device)[action.index]
        network_input = torch.cat((parent_hidden_state.squeeze(0), action_one_hot)).unsqueeze(0)

        next_hidden_state = self.dynamics_network(network_input)
        reward = self.reward_network(network_input)
        value = self.value_network(next_hidden_state)
        policy = self.policy_network(next_hidden_state)

        return NetworkOutput(
            value,
            reward,
            {Action(a): policy[0, a].item() for a in range(policy.shape[1])},
            policy,
            next_hidden_state
        )
