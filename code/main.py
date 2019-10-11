from collections import namedtuple

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count, accumulate
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from train import run_episodes
from model import ReplayMemory, DQN

"""Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""


# TODO:
# Add option for replay
def main(args):
    # experiment params
    env = args.env
    replay_bool = args.replay
    target_bool = args.fixed_T_policy
    reward_num = args.reward_clip
    seed = args.seed
    #layers = args.architecture
    output_file = args.output

    # Get architecture
    architecture_dict = {0 : [64, 64], 1: [256, 256], 2: [128, 64, 32]}
    layers = architecture_dict[args.architecture]

    # training params
    num_episodes = args.n_episodes
    lr = args.lr
    batch_size = args.batch_size
    discount_factor = args.gamma
    target_update = args.target_update_every

    # misc
    if not replay_bool:
        memory_size = 10000
    else:
        memory_size = 1
        batch_size = 1

    # if target network is the same, then update every time
    if not target_bool:
        target_update = 0

    env_options = {
        0: None,
        1: gym.make('MountainCar-v0'),
        2: gym.make('BipedalWalker-v2'),
        3: gym.make('Acrobot-v1'),
        4: gym.make('CartPole-v0'),

    }

    assert env in env_options.keys()

    env = env_options[env]

    # seed
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = args.device

    # NOT SURE IF NECESSARY
    # if device == "cuda":
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get number of features from gym environment
    obs = env.reset()
    num_features = obs.size

    # models
    model = DQN(num_features, n_actions, layers).to(device)
    target_net = DQN(num_features, n_actions, layers).to(device)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    memory = ReplayMemory(memory_size)

    episode_durations, losses, rewards = run_episodes(model, target_net, memory, env, num_episodes, batch_size, discount_factor, optimizer,
                                             target_update=target_update, replay_bool=replay_bool,
                                             reward_clip=reward_num, device = device)

    print('Complete')

    # Create results directory if it does not exist yet
    os.makedirs("results", exist_ok=True)

    # Print results to CSV file
    with open(f"results/{output_file}.csv", 'w') as f:
        f.write("EPISODE;DURATION;LOSSES\n")
        for i, (duration, losses) in enumerate(zip(episode_durations, losses)):
            f.write(f"{i};{duration};{losses}\n")

    with open(f"results/{output_file}_cumm_rewards.csv", 'w') as f:
        for episode in rewards:
            comma=""
            for reward in list(accumulate(episode)):
                f.write(f"{comma}{reward}")
                comma=";"
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    experiment_parse = parser.add_argument_group('Experiments')
    experiment_parse.add_argument(
        '--env', type=int, required=True, default=1,
        help='Which environment to use; 0:Baird, 1:Mountain Car, 2:Bipedal Walker, 3:Hanging Joints, 4:Pole Balancing.')

    experiment_parse.add_argument(
        '--replay', type=int, required=True, default=1,
        help='Basically bool: 1 if the training should involve Experience Replay, else 0')

    experiment_parse.add_argument(
        '--fixed_T_policy', type=int, required=True, default=1,
        help='Basically bool: 1 if should use fixed target policy, else 0')

    experiment_parse.add_argument(
        '--reward_clip', type=int, required=True, default=1,
        help='Which reward clipping to use; 0: None, 1: [-1,1] R: reward*R')

    experiment_parse.add_argument(
        '--seed', type=int, required=True, default=42,
        help='Random seed')

    experiment_parse.add_argument(
        '--architecture', type=int, required=True, default=1,
        help='Which architecture to use: 0 : [64, 64]; 1: [256, 256] ; 2:[128, 64, 32]')

    experiment_parse.add_argument(
        '--output', type=str, required=True,
        help='Filename of results output file without extension')

    training_parse = parser.add_argument_group('Training')

    training_parse.add_argument(
        '--n_episodes', type=int, default=100,
        help="number of episodes")

    training_parse.add_argument(
        '--batch_size', type=int, default=32,
        help="batch size")

    training_parse.add_argument(
        '--lr', type=float, default=0.001,
        help="learning rate")

    training_parse.add_argument(
        '--device', type=str, default="cuda:0",
        help="Training device 'cpu' or 'cuda:0'")

    training_parse.add_argument(
        '--target_update_every', type=int, default=10,
        help="Update target every XX episodes")

    training_parse.add_argument(
        '--gamma', type=float, default=0.999,
        help="discount factor")

    args = parser.parse_args()
    main(args)

