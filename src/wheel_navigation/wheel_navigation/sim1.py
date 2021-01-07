#!/usr/bin/env python3

import rclpy

import torch

from wheel_navigation.env import Environment

class Brain(torch.nn.Module):
    """Pytorch neural network model"""

    ACTION = torch.tensor([[[ 1., 1.]], [[ 1., 0.]], [[ 1., -1.]],
                           [[ 0., 1.]], [[ 0., 0.]], [[ 0., -1.]],
                           [[-1., 1.]], [[-1., 0.]], [[-1., -1.]]])
    
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

    def get_actions(self, obs):
        act = torch.empty(0, 2)
        logits = self.forward(obs)
        for logit in logits:
            idx = torch.distributions.categorical.Categorical(logits=logit).sample().item()
            act = torch.cat([act, Brain.ACTION[idx]], dim=0)
        return act

def main(args=None):
    rclpy.init(args=args)
    node = Environment(Brain(40, 9))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()