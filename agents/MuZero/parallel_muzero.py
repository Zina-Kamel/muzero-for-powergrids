import multiprocessing as mp
import time
import grid2op
from lightsim2grid import LightSimBackend
from muzero_agent import MuZero, generate_rollout, train_model
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from muzero_config import MuZeroConfig
from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace
from grid2op.Reward import L2RPNReward
import wandb
import torch
import numpy as np



def self_play_worker(shared_storage, queue, worker_id, training_step):
    
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend(), reward_class=L2RPNReward)
    gym_env = GymEnv(env)

    p = gym_env.init_env.parameters
    p.MAX_LINE_STATUS_CHANGED = 1
    p.MAX_SUB_CHANGED = 1
    gym_env.init_env.change_parameters(p)
    gym_env.action_space = DiscreteActSpace(env.action_space, attr_to_keep=["set_bus", "change_bus", "set_line_status", "change_line_status"])
    gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space)
    
    config = MuZeroConfig(
    batch_size=128, 
    window_size= 5000, 
    training_iterations=1000000,
    discount=0.9,
    action_dim=gym_env.action_space.n,
    hidden_layer_size=256,
    input_size= gym_env.observation_space.shape[0],
    observation_space_size= gym_env.observation_space.shape[0],
    max_rollout_steps=1000,
    dirichlet_alpha=0.3,
    num_simulations=50,
    lr_init=1e-4,
    lr_decay_steps=10000,
    visualize = False,
    visualize_frequency=1,
    visualize_path="visualization/MuZero/",
    exploit_threshold=200,)

    agent = MuZero(gym_env, config, shared_storage)
    
    def detach_rollout(rollout):
        for idx, obs in enumerate(rollout.observations):
            if isinstance(obs, torch.Tensor):
                rollout.observations[idx] = obs.detach().cpu().numpy()
        for idx, rew in enumerate(rollout.rewards):
            if isinstance(rew, torch.Tensor):
                rollout.rewards[idx] = rew.detach().cpu().numpy()
        for idx, act in enumerate(rollout.history):
            if isinstance(act, torch.Tensor):
                rollout.history[idx] = act.detach().cpu().numpy()
        for idx, rootval in enumerate(rollout.root_values):
            if isinstance(rootval, torch.Tensor):
                rollout.root_values[idx] = rootval.detach().cpu().numpy()
        if hasattr(rollout, 'stats') and rollout.stats:
            for key, val in rollout.stats.items():
                if isinstance(val, torch.Tensor):
                    rollout.stats[key] = val.detach().cpu().numpy()
        return rollout

    while True:
        network = shared_storage.get_last_network()
        rollout = generate_rollout(agent, config, network, training_step)
        rollout = detach_rollout(rollout)
        queue.put(rollout)
        print(f"Worker {worker_id} collected a rollout")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    wandb.init(
        project="ML4Power",
        entity="florence-cloutier-mila",
        name="parallel-muzero-l2rpnReward"
    )
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend(), reward_class=L2RPNReward)
    gym_env = GymEnv(env)

    p = gym_env.init_env.parameters
    p.MAX_LINE_STATUS_CHANGED = 1
    p.MAX_SUB_CHANGED = 1
    gym_env.init_env.change_parameters(p)
    gym_env.action_space = DiscreteActSpace(env.action_space, attr_to_keep=["set_bus", "change_bus", "set_line_status", "change_line_status"])
    gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space)
    
    config = MuZeroConfig(
    batch_size=128, 
    window_size= 50, 
    training_iterations=1000000,
    discount=0.9,
    action_dim=gym_env.action_space.n,
    hidden_layer_size=256,
    input_size= gym_env.observation_space.shape[0],
    observation_space_size= gym_env.observation_space.shape[0],
    max_rollout_steps=1000,
    dirichlet_alpha=0.3,
    num_simulations=50,
    lr_init=1e-4,
    lr_decay_steps=10000,
    visualize = False,
    visualize_frequency=1,
    visualize_path="visualization/MuZero/",
    exploit_threshold=200,)
 
    shared_storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)
    rollout_queue = mp.Queue(maxsize=100)

    num_workers = 4
    workers = []
    training_step = 0
    for worker_id in range(num_workers):
        p = mp.Process(target=self_play_worker, args=(shared_storage, rollout_queue, worker_id, training_step))
        p.start()
        workers.append(p)

    
    while True:
        while not rollout_queue.empty():
            rollout = rollout_queue.get()
            replay_buffer.save_rollout(rollout)

        if len(replay_buffer.buffer) >= 50:
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_dim)
            loss, val_loss, reward_loss, pol_loss = train_model(config, shared_storage, replay_buffer, training_step)
            
            wandb.log({
                "Average Reward": np.mean([r.total_rewards() for r in replay_buffer.buffer]),
                "Total Loss": loss,
                "Value Loss": val_loss,
                "Reward Loss": reward_loss,
                "Policy Loss": pol_loss,
                "Training Step": training_step
            }, step=training_step)
            print(f"Trained on a batch at step {training_step}")
            training_step += 1

        time.sleep(1)
