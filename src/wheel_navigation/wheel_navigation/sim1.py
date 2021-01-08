#!/usr/bin/env python3

import rclpy

import torch

from wheel_navigation.env import Environment

class Brain(torch.nn.Module):
    """Pytorch neural network model"""

    ACTION = [[ 1., 1.], [ 1., 0.], [ 1., -1.],
              [ 0., 1.], [ 0., 0.], [ 0., -1.],
              [-1., 1.], [-1., 0.], [-1., -1.]]
    
    def __init__(self, num_input, num_output, num_hidden=40):
        self.num_output = num_output
        
        super(Brain, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs):
        policy = self.get_policy(obs)
        act_idx = [p.sample().item() for p in policy]
        act = [Brain.ACTION[idx] for idx in act_idx]
        return torch.tensor(act)
    
    def get_policy(self, obs):
        logits = self.forward(obs)
        return [torch.distributions.categorical.Categorical(logits=logit) for logit in logits]

def main(args=None):
    rclpy.init(args=args)
    node = Environment(Brain(40, 9))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()