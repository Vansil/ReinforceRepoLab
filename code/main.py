from collections import namedtuple

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import argparse

from model import ReplayMemory, DQN

"""Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""



def get_epsilon(it, p = 0.05, when_to_end = 1000):
    if it>=when_to_end:
        return p
    else:
        return 1-(it)/(when_to_end*(1+p))
def select_action(model, state, epsilon):
    state = torch.tensor(state).to(dtype = torch.float)
    scores = model(state)
    action = int(np.random.rand() * 2) if np.random.rand() < epsilon else torch.argmax(scores).item()
    return action

def compute_q_val(model, state, action):
    scores = model(state)
    q_val = torch.gather(scores, 1 , action.unsqueeze(1))
    return q_val

def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    scores = model(next_state)
    next_q = torch.max(scores, 1)[0]
    target = reward+ (torch.ones(done.size())-done.to(dtype=torch.float)) *discount_factor*next_q
    target = target.unsqueeze(1)
    return target

def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())
def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):

    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    losses = []
    for i in range(num_episodes):
        #initialise inside the episode counter
        episode_steps=0
        #initialise epsilon
        epsilon = get_epsilon(global_steps)
        #start the environment
        s = env.reset()
        action = select_action(model, s, epsilon)
        while True:
            s_new, r, is_terminal, prob  = env.step(action)
            #store the state
            memory.push((s, action, r, s_new, is_terminal))
            #train the Qnet
            loss = train(model, memory, optimizer, batch_size, discount_factor)
            losses.append(loss)
            #update counters
            episode_steps+=1
            global_steps+=1
            #take new action
            epsilon = get_epsilon(global_steps)
            action = select_action(model, s_new, epsilon)
            s = s_new
            if is_terminal:
                episode_durations.append(episode_steps)
                break

    return episode_durations, losses
# TODO:
# Add option for replay
# Add option for Fixed Target Policy
# Add option for type of Reward Clipping
# Add option for other (hyper) parameters
def main(args):


    batch_size = 128
    discount_factor = 0.999
    target_update = 10

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))



    env_options = {
        0 : None,
        4 : gym.make('CartPole-v0').unwrapped,
    }
    assert args.env in env_options

    env = env_options[args.env]

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    model = DQN(screen_height, screen_width, n_actions).to(device)
    target = DQN(screen_height, screen_width, n_actions).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0

    episode_durations = []

    num_episodes = 100
    episode_durations, losses= run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)


    print('Complete')

    plt.ioff()
    plt.show()

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=int,
                        required=True,
                        help='Which environment to use; 0:Baird, 1:Mountain Car, 2:Bipedal Walker, 3:Hanging Joints, 4:Pole Balancing'
                        )
    args = parser.parse_args()
    main(args)