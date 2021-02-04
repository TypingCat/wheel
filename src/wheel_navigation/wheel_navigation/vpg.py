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

class VPG(Node):
    """Vanilla Policy Gradient"""

    def __init__(self):
        super().__init__('wheel_navigation_vpg')
        self.actor = MLP(num_input=40, num_output=2)
        self.critic = MLP(num_input=40, num_output=1)
        self.policy_std = torch.exp(torch.tensor([-0.5, -0.5]))
        self.batch = Batch()
        self.batch_size_max = 100
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)
        print(self.actor)
        
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
            mu = self.actor(obs)
            policy = torch.distributions.normal.Normal(mu, self.policy_std)
            act = policy.sample()
        self.unity.set_command(act)
        self.batch.store(exp, act)
        
        # Start learning
        print(f'\rBatch: {self.batch.size()}/{self.batch_size_max}', end='')
        if self.batch.size() >= self.batch_size_max:
            loss = self.learning(self.batch.pop())
            print()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self, batch):
        """Training neural network using batch data"""
        for episode in batch:    # Replace rewards with weights
            episode[:, 40:41] = sum(episode[:, -1]) * torch.ones(episode.shape[0], 1)
        data = torch.cat(batch, dim=0)

        # Calculate loss
        mu = self.actor(data[:, 0:40])
        policy = torch.distributions.normal.Normal(mu, self.policy_std)
        logp = policy.log_prob(data[:, 41:]).sum(axis=1)
        weight = data[:, 40:41]
        loss = -(logp * weight).mean()

        # Optimize network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def main(args=None):
    rclpy.init(args=args)
    node = VPG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()