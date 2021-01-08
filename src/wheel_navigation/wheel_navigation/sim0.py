#!/usr/bin/env python3

import rclpy

import torch

from wheel_navigation.env import Environment

class Brain(torch.nn.Module):
    """Pytorch neural network model"""
    
    def __init__(self, num_input, num_output, num_hidden=40):
        super(Brain, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs):
        return self.forward(obs)

def main(args=None):
    rclpy.init(args=args)
    node = Environment(Brain(40, 2))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()