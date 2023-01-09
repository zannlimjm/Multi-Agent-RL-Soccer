from arguments import get_args
from ppo_agent import ppo_agent
import gfootball.env as football_env
import numpy as np
from models import cnn_net
import torch
import os

# get the tensors
def get_tensors(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)

if __name__ == '__main__':
    args = get_args()
    model_path = args.save_dir + args.env_name + '/final_pass_shoot_keeper.pt'
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True,render=True)
    network = cnn_net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # start to do the test
    obs = env.reset()
    num_episode = 0
    total_reward = 0.0
    total_step = 0

    while num_episode <= 1000: #30 episodes
        obs_tensor = get_tensors(np.expand_dims(obs, 0))
        with torch.no_grad():
            _, pi = network(obs_tensor)
        actions = torch.argmax(pi, dim=1).item()
        obs, reward, done, _ = env.step(actions)
        total_reward += reward
        total_step += 1
        if done:
            num_episode += 1
            obs = env.reset()
    print('After {} episodes, the average reward is {}, average step is {}'.format(num_episode-1,total_reward/(num_episode-1),total_step/(num_episode-1)))
    env.close()
