
##python3 -m gfootball.play_game --players "A3C:left_players=1;keyboard:right_players=1" --level academy_pass_and_shoot_with_keeper


from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
from gfootball.env import player_base
from gfootball.examples import models  
import gym
import joblib
import numpy as np
from gfootball.env.players.A3Cmodel import ActorCritic
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import gfootball.env as football_env


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)
    env = football_env.create_environment(env_name='academy_pass_and_shoot_with_keeper',
                                      stacked=True,
                                      representation='extracted',
                                      render=True)
    self.num_states = 16
    self.num_actions = env.action_space.n
    self.model = ActorCritic(self.num_states, self.num_actions)
    self.model.load_state_dict(torch.load("/home/sz/football/A3C-google-research-football/trained_models/passShoot_train_work.pth", map_location=torch.device('cpu')))
    self.obs = env.reset()
    self.h_0 = torch.zeros((1, 512), dtype=torch.float)
    self.c_0 = torch.zeros((1, 512), dtype=torch.float)
    env.reset()

  def _get_tensors(self,obs):
    obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    # decide if put the tensor on the GPU
    if torch.cuda.is_available():
        obs_tensor = obs_tensor.cuda()
    return obs_tensor

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'
    obs_tensor = self._get_tensors(self.obs)
    logits, value, self.h_0, self.c_0 = self.model(obs_tensor, self.h_0, self.c_0)
    policy = F.softmax(logits, dim=1)
    action = torch.argmax(policy).item()
    action = int(action)
    return action

