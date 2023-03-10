import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_train
import torch.multiprocessing as _mp
import shutil
import gfootball.env as football_env


def get_args():
    parser = argparse.ArgumentParser("A3C for google research football")
    parser.add_argument("--env_name", type=str, default='11_vs_11_easy_stochastic')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--lr_decay",type=bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.987, help='discount factor for rewards')
    parser.add_argument("--tau", type=float, default=1.0, help='parameter for GAE')
    parser.add_argument("--beta", type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=128)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of episode between savings")
    parser.add_argument("--print_interval",type=int,default=50)
    parser.add_argument("--saved_path", type=str, default="/home/sz/football/A3C-google-research-football/trained_models")
    parser.add_argument("--saved_path_file",type=str, default="/home/sz/football/A3C-google-research-football/trained_models/11v11_fullmodel.pth")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    
    parser.add_argument("--use_gpu", type=bool, default=False)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    env = football_env.create_environment(env_name=opt.env_name,
                                      stacked=True,
                                      representation='extracted',
                                      render=False)
    num_states = 16
    num_actions = env.action_space.n
    global_model = ActorCritic(num_states, num_actions)
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()
    if opt.load_from_previous_stage:
        if os.path.isfile(opt.saved_path_file):
            if opt.use_gpu:
              global_model.load_state_dict(torch.load(opt.saved_path_file))
              global_model.eval()
            else:
              global_model.load_state_dict(torch.load(opt.saved_path_file,map_location=torch.device('cpu')))
              global_model.eval()

    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr, eps=opt.eps)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
