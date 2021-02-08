#!/usr/bin/env python3

import torch
import time

import rclpy
from rclpy.node import Node

from wheel_navigation.env import Unity, Batch

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

class SPG(Node):
    """Simple Policy Gradient"""

    def __init__(self):
        super().__init__('wheel_navigation_spg')

        # Set parameters
        self.policy_std = torch.exp(torch.tensor([-0.5, -0.5]))
        self.batch_size_max = 500

        # Initialize network
        self.brain = MLP(num_input=40, num_output=2)
        self.batch = Batch()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)
        print(self.brain)
        
        # Connect to Unity
        self.get_logger().warning("Wait for Unity scene play")
        self.unity = Unity()
        self.get_logger().info("Unity environment connected")

        # Initialize ROS
        self.get_logger().info("Start simulation")
        self.time_start = time.time()
        self.timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        # Simulate environment one-step forward
        self.unity.env.step()
        exp = self.unity.get_experience()
        if not exp: return

        # Set commands
        obs = torch.cat([exp[agent]['obs'] for agent in exp], dim=0)
        with torch.no_grad():
            mu = self.brain(obs)
            policy = torch.distributions.normal.Normal(mu, self.policy_std)
            act = policy.sample()
        self.unity.set_command(act)
        self.batch.store(exp, act.unsqueeze(1))
        
        # Start learning
        print(f'\rBatch: {self.batch.size()}/{self.batch_size_max}', end='')
        if self.batch.size() >= self.batch_size_max:
            loss = self.learning(self.batch.pop())
            print()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self, batch):
        """Training the brain using batch data"""
        
        # Replace reward with reward_sum
        for episode in batch:
            episode[:, 40:41] = sum(episode[:, -1]) * torch.ones(episode.shape[0], 1)
        
        # Optimize the brain
        data = torch.cat(batch, dim=0)
        mu = self.brain(data[:, 0:40])
        act, reward_sum = data[:, 41:], data[:, 40:41]

        policy = torch.distributions.normal.Normal(mu, self.policy_std)        
        logp = policy.log_prob(act).sum(axis=1)        
        loss = -(logp * reward_sum).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def main(args=None):
    rclpy.init(args=args)
    node = SPG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()