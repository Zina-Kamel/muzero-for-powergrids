from rollout import Rollout

# fixed parameters are taken from MuZero Paper
class MuZeroConfig(object):
    def __init__(self,
                    batch_size: int,
                    window_size: int,
                    action_dim: int,
                    hidden_layer_size:int,
                    training_iterations: int,
                    discount: float,
                    lr_init: float,
                    lr_decay_steps: float,
                    observation_space_size: int,
                    input_size: int,
                    max_rollout_steps: int,
                    dirichlet_alpha: float,
                    num_simulations: int,   
                    visualize: bool,
                    visualize_frequency: int,
                    visualize_path: str, 
                    exploit_threshold: float,                
                    ):
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.discount = discount
        self.action_dim = action_dim
        self.hidden_layer_size = hidden_layer_size
        self.training_iterations = training_iterations
        self.observation_space_size = observation_space_size
        self.input_size = input_size
        self.max_rollout_steps = max_rollout_steps
        self.dirichlet_alpha = dirichlet_alpha
        self.num_simulations = num_simulations
        self.training_steps = int(1000e3)
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.num_unroll_steps = 5
        self.td_steps = 7
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
        self.visualize = visualize
        self.visualize_frequency = visualize_frequency
        self.visualize_path = visualize_path
        self.exploit_threshold = exploit_threshold
    
    def new_rollout(self, env):
        return Rollout(self.action_dim, self.discount, env)