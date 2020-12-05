#!/usr/bin/env python3

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    """Pytorch neural network model"""
    
    def __init__(self, num_input, num_output):
        super(Brain, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    brain = Brain(1004, 2)
    print("\nModel:")
    print(brain)

    obs = torch.randn(3, 1004)
    print("\nRandom input:")
    print(obs)

    act = brain(obs)
    print("\nResult:")
    print(act)
