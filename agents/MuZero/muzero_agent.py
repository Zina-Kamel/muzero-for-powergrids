import os
import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace
from lightsim2grid import LightSimBackend
import networkx as nx
from muzero_config import MuZeroConfig
from replay_buffer import ReplayBuffer
from node import Node
import numpy as np
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from minmax_stats import MinMaxStats
from shared_storage import SharedStorage
from typing import List
import torch.optim as optim
from action import Action
import cv2
import wandb
from grid2op.PlotGrid import PlotMatplot
from grid2op.Reward import EpisodeDurationReward
from grid2op.Reward import IncreasingFlatReward
from grid2op.Reward import L2RPNReward
from grid2op.Reward import DistanceReward
from grid2op.Reward import LinesCapacityReward, FlatReward, CombinedScaledReward
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from visualising_grid import create_visualisation_gif
sys.path.remove(root_dir)

class MuZero:
    def __init__(self, env, config: MuZeroConfig):
        self.env = env
        self.replay_buffer = ReplayBuffer(config)
        self.shared_storage = SharedStorage(config)
        
        rewards = []
        losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        for i in range(config.training_iterations):
            collect_rollout(self, config, self.shared_storage, self.replay_buffer, i)

            last_rollout = self.replay_buffer.last_rollout()
            survival_rate = last_rollout.episode_length / config.max_rollout_steps
            total_rewards = last_rollout.total_rewards() 
            rewards.append(total_rewards)
                    
            print('Iteration ' + str(i+1) + ' ' + 'reward: ' + str(total_rewards))
            
            loss, val_loss, reward_loss, pol_loss = train_model(config, self.shared_storage, self.replay_buffer, i)
                        
            print('Iteration ' + str(i+1) + ' ' + 'Loss: ' + str(loss))
            losses.append(loss)
            
            wandb.log({"loss": loss, "charts/cum_reward": sum(rewards), "Value Network Loss": val_loss, "Reward Network Loss": reward_loss, "Policy Network Loss": pol_loss, f"Episode Survival, Maximum Steps {config.max_rollout_steps}": survival_rate, "Reward": total_rewards, "charts/episodic_length": last_rollout.episode_length}, step=i)
    

def compute_loss(pred, target):
    return F.mse_loss(pred, target)

def modify_gradient(tensor, factor: float):
    return tensor * factor + tensor.detach() * (1 - factor)

def optimize_parameters(optimizer: torch.optim.Optimizer, model, batch_data, reg_factor: float):
    if not batch_data:
        return 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    optimizer.zero_grad()
    total_loss = 0.0

    for obs, act_seq, target_values in batch_data:
        obs = torch.tensor(obs, dtype=torch.float32).to(device) if isinstance(obs, np.ndarray) else obs.to(device)
        if isinstance(act_seq, list):
            act_seq = torch.tensor(act_seq, dtype=torch.long).to(device)
        elif isinstance(act_seq, np.ndarray):
            act_seq = torch.tensor(act_seq, dtype=torch.long).to(device)
        else:
            act_seq = act_seq.to(device)
        target_values = [(
            torch.tensor(tv[0], dtype=torch.float32).to(device),  
            torch.tensor(tv[1], dtype=torch.float32).to(device),
            torch.tensor(tv[2], dtype=torch.float32).to(device) if tv[2] is not None else None
        ) for tv in target_values]


        # Initial inference
        val, rew, _, pol, state = model.initial_inference(obs)
        outputs = [(1.0, val, rew, pol)]

        # Recurrent inference
        for act in act_seq:
            val, rew, _, pol, state = model.recurrent_inference(state, Action(act))
            outputs.append((1.0 / len(act_seq), val[0], rew[0], pol[0]))
            state = modify_gradient(state, 0.5)
            
        value_network_losses = []
        rewards_network_losses = []
        policy_network_losses = []
        

        # Compute losses
        for idx, ((val, rew, pol), (target_val, target_rew, target_pol)) in enumerate(zip(outputs, target_values)):
            # Value and reward losses
            val = val.squeeze()
            target_val = target_val.squeeze()

            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            if not isinstance(target_val, torch.Tensor):
                target_val = torch.tensor(target_val, dtype=torch.float32)

            val = val.to(device)
            target_val = target_val.to(device)

            val = val.view_as(target_val)

            loss_val = torch.nn.functional.smooth_l1_loss(target_val, val) 
        

            if idx > 0:
                rew = rew.squeeze()
                target_rew = target_rew.squeeze()

                if not isinstance(rew, torch.Tensor):
                    rew = torch.tensor(rew, dtype=torch.float32)
                if not isinstance(target_rew, torch.Tensor):
                    target_rew = torch.tensor(target_rew, dtype=torch.float32)

                rew = rew.to(device)
                target_rew = target_rew.to(device)

                rew = rew.view_as(target_rew)

                loss_rew = torch.nn.functional.smooth_l1_loss(target_rew, rew) 
            else:
                loss_rew = torch.tensor(0.0, dtype=torch.float32).to(device)


            # Policy loss (cross-entropy with logits)
            if target_pol is not None and target_pol.numel() > 0:
                
                probs = pol  # already softmaxed

                eps = 1e-8
                log_probs = torch.log(probs + eps)

                # Normalize target
                target_probs = target_pol / target_pol.sum()

                # Compute soft label cross-entropy
                loss_pol = torch.sum(-target_probs * log_probs, dim=-1).mean()
            else:
                loss_pol = torch.tensor(0.0, dtype=torch.float32).to(device)

            value_network_losses.append(loss_val.item())
            rewards_network_losses.append(loss_rew.item())
            policy_network_losses.append(loss_pol.item())

            prediction_loss =  loss_val + loss_rew + loss_pol
            total_loss += prediction_loss


    # Average loss over batch
    total_loss /= len(batch_data)
    value_network_losses = np.mean(value_network_losses)
    rewards_network_losses = np.mean(rewards_network_losses)
    policy_network_losses = np.mean(policy_network_losses)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), value_network_losses, rewards_network_losses, policy_network_losses   


def train_model(config: MuZeroConfig, storage: SharedStorage, buffer: ReplayBuffer, step_count: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = storage.get_last_network().to(device)  
    lr = config.lr_init 
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    batch_data = buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_dim)
    loss = optimize_parameters(opt, model, batch_data, config.weight_decay)
    model.total_training_steps += 1
    return loss


def apply_exploration_noise(config: MuZeroConfig, node: Node):
    available_actions = list(node.children.keys())
    noise_vals = np.random.dirichlet([config.dirichlet_alpha] * len(available_actions))
    mix_ratio = config.root_exploration_fraction
    for action, noise in zip(available_actions, noise_vals):
        node.children[action].prior = node.children[action].prior * (1 - mix_ratio) + noise * mix_ratio

def generate_node_children_actions(node, net_output):
    node.hidden_state = net_output.hidden_state
    node.reward = net_output.reward
    for action, prob in net_output.policy_logits.items():
        node.children[action] = Node(prob)
        
        
def execute_mcts(config, root, action_log, model):
    stats = MinMaxStats(root.value(), root.value())
    
    for _ in range(config.num_simulations):
        action_record = action_log.clone()
        node = root
        path = [node]
        
        # traverse according to UCB score until leaf node is reached
        while node.expanded():
            chosen_action, node = pick_highest_ucb_child(config, node, stats)
            action_record.add_action(chosen_action)
            path.append(node)
        
        # expand selected leaf node
        parent_node = path[-2]
        output = model.recurrent_inference(parent_node.hidden_state, action_record.last_action())
        generate_node_children_actions(node, output)
        
        # propagate the value approximated by the predicted network to all the nodes in the search path
        propagate_values(path, output.value, config.discount, stats)

def pick_highest_ucb_child(config: MuZeroConfig, node: Node, stats: MinMaxStats):
    _, action, child_node = max(
        (compute_ucb(config, node, child, stats), action, child) for action, child in node.children.items()
    )
    return Action(action) if isinstance(action, int) else action, child_node

def compute_ucb(config: MuZeroConfig, parent: Node, child: Node, stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_component = pb_c * child.prior
    value_component = stats.normalize(child.reward + config.discount * child.value()) if child.visit_count > 0 else 0
    return prior_component + value_component

def propagate_values(path: List[Node], value: float, discount: float, stats: MinMaxStats):
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        stats.update(node.value())
        value = node.reward + discount * value

def select_action(config, node, training_steps):
    # actions are selected probabilistically based on their visit counts
  visits_dist = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  
  temperature = (
        3 if training_steps < 50 else
        2 if training_steps < 70 else
        1 if training_steps < 80 else
        0.5 if training_steps < 90 else
        0.25 if training_steps < 100 else
        0.125 if training_steps < 110 else
        0.0625 
    )
  
  visits = np.array([visits for visits, _ in visits_dist])
  visit_exp = np.exp(visits)
  policy = visit_exp / np.sum(visit_exp)
  policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
  action_index = np.random.choice(range(len(policy)), p=policy)
    
  return action_index

def generate_rollout(agent, config, network, training_step):
    rollout = config.new_rollout(agent.env)
    obs = agent.env.reset()
    obs_raw, _ = obs  
    obs = obs_raw
    rollout.add_observation(obs)
    plot_helper = PlotMatplot(agent.env.init_env.observation_space)
    exploit_mode = False
    
    while not rollout.done and len(rollout.history) < config.max_rollout_steps:
        if len(rollout.history) % 10==0:
            print(f"Rollout step: {len(rollout.history)}")
        obs = torch.tensor(rollout.get_observation(-1)).to(agent.device) # latest observation in this rollout (this will be the root of MCTS)
        root = Node(0)
        network_output_inital_obs = network.initial_inference(obs) # returns value, reward, action probabilities, policy, hidden state representation of the obs
        # before applying MTCS, we need to generate the children of the root node
        # based on the initial observation
        generate_node_children_actions(root, network_output_inital_obs) 
        apply_exploration_noise(config, root)
        
        execute_mcts(config, root, rollout.get_action_history(), network)
        # if config.visualize and training_step % config.visualize_frequency == 0:
        #     tree_path = config.visualize_path + f"rollout_{training_step}/step_{len(rollout.history)}/mcts_tree.png"
        #     os.makedirs(os.path.dirname(tree_path), exist_ok=True)
        #     plot_mcts_tree(root, save_path=tree_path)

        action = select_action(config, root, training_step)

        next_obs, infos, done = rollout.take_action(action)
        if not done and config.visualize and training_step % config.visualize_frequency == 0:
            visualize_path = config.visualize_path + f"iteration_{training_step}/step_{len(rollout.history)}.png"
            next_obs_grid2op = agent.env.init_env.get_obs()
            cur_grid = plot_helper.plot_obs(next_obs_grid2op)
            visualize_dir = os.path.dirname(visualize_path)
            os.makedirs(visualize_dir, exist_ok=True)
            cur_grid.savefig(visualize_path)
            plt.close(cur_grid)
            
        rollout.store_stats(root)
    
    rollout.episode_length = len(rollout.history)
    
    return rollout
        

# retrieves the latest network and use it to add a new rollout to the buffer
def collect_rollout(agent, config: MuZeroConfig, shared_storage: SharedStorage,
                 replay_buffer: ReplayBuffer, training_step):
    network = shared_storage.get_last_network()
    rollout = generate_rollout(agent, config, network, training_step)
    replay_buffer.save_rollout(rollout)
    if config.visualize and training_step % config.visualize_frequency == 0:
        visualizations_path = config.visualize_path + f"iteration_{training_step}/"
        create_visualisation_gif(visualizations_path)

wandb.init(
    project="ML4Power",
    entity="florence-cloutier-mila",
    name="muzero-l2rpn",  
)

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=L2RPNReward)
example_obs = env.reset()

gym_env = GymEnv(env)

# From RL2Grid
# Making sure we can act on 1 sub / line status at the same step
p = gym_env.init_env.parameters
p.MAX_LINE_STATUS_CHANGED = 1 
p.MAX_SUB_CHANGED = 1 
gym_env.init_env.change_parameters(p)

# Set the action space
gym_env.action_space = DiscreteActSpace(env.action_space,
                                        attr_to_keep=["set_bus",
                                                      "change_bus",
                                                      "set_line_status",
                                                      "change_line_status",
                                                      ]
                                        )
# Set the observation space
gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space,
                                    # attr_to_keep=obs_attrs,
                                    #divide={"gen_p": gym_env.init_env.gen_pmax,
                                    #        "actual_dispatch": gym_env.init_env.gen_pmax},
)

example_gym_obs = gym_env.reset()


config = MuZeroConfig(
    batch_size=128, 
    window_size= 1000, 
    training_iterations=100000,
    discount=0.9,
    action_dim=gym_env.action_space.n,
    hidden_layer_size=256,
    input_size= gym_env.observation_space.shape[0],
    observation_space_size= gym_env.observation_space.shape[0],
    max_rollout_steps=100,
    dirichlet_alpha=0.3,
    num_simulations=75,
    lr_init=1e-4,
    lr_decay_steps=10000,
    visualize = False,
    visualize_frequency=20,
    visualize_path="visualization/MuZero/",
    exploit_threshold=200,)
    
muzero_agent = MuZero(gym_env, config)