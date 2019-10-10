from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        """Saves a transition."""
        self.memory.append(transition)
        if len(self.memory)>self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, state_features, num_actions, layers = [64,64]):
        super(DQN, self).__init__()
        self.net = nn.Sequential()
        for i,layer in enumerate(layers):
            if i==0:
                self.net.add_module("Input",nn.Linear(state_features, layer))
                self.net.add_module(f"ReLU{i}", nn.ReLU())
            elif i==len(layers)-1:
                self.net.add_module(f"Hidden{i}",nn.Linear(layer, num_actions))
            else:
                self.net.add_module("Output",nn.Linear(layer, layer))
                self.net.add_module(f"ReLU{i}", nn.ReLU())

    def forward(self, x):
        return self.classifier(x)

