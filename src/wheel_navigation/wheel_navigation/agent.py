#!/usr/bin/env python3

import torch

class MLP(torch.nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, num_input, num_output, num_hidden=40):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
class Batch:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.temp = {}
        self.data = []

    def size(self):
        return sum([d.shape[0] for d in self.data])

    def store(self, exp, metadata):
        """Save experiences and wrap finished episodes"""
        # sample[ 0:40] = observation
        # sample[40:41] = reward
        # sample[41:  ] = metadata
        for i, agent in enumerate(exp):
            sample = torch.cat([exp[agent]['obs'], exp[agent]['reward'], metadata[i, :].unsqueeze(0)], dim=1)
            if agent not in self.temp.keys():
                self.temp[agent] = sample
            else:
                self.temp[agent] = torch.cat([self.temp[agent], sample], dim=0)
            if exp[agent]['done']:
                self.data.append(self.temp[agent])
                del self.temp[agent]
    
    def pop(self):
        episodes = self.data
        self.data = []
        return episodes

def discount_cumsum(arr, discount_factor):
    for i in range(len(arr)-2, -1, -1):
        arr[i] += discount_factor * arr[i+1]
    return arr