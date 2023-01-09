import torch
from src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
# from tensorboardX import SummaryWriter
import timeit
import gfootball.env as football_env
import numpy as np
from datetime import datetime
env = football_env.create_environment(env_name='academy_pass_and_shoot_with_keeper',
                                                                stacked=True,
                                                                representation='extracted',
                                                                render=False)
num_states = 16
num_actions = env.action_space.n
model = ActorCritic(num_states, num_actions)
for layers in model.children():
    if hasattr(layers, 'reset_parameters'):
        layers.reset_parameters()