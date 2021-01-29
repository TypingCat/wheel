#!/usr/bin/env python3

import torch
import time

import rclpy
from rclpy.node import Node

from wheel_navigation.env import Unity, Batch

class Brain(torch.nn.Module):
    """Pytorch neural network model"""
    ACTION = [[ 1., 1.], [ 1., 0.], [ 1., -1.],
              [ 0., 1.], [ 0., 0.], [ 0., -1.],
              [-1., 1.], [-1., 0.], [-1., -1.]]
    
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

class SPG(Node):
    """Simple Policy Gradient"""

    def __init__(self):
        super().__init__('wheel_navigation_spg')
        self.brain = Brain(num_input=40, num_output=9)
        self.batch = Batch()
        self.batch_size_max = 100
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.02)
        print(self.brain)
        
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
            logit = self.brain(obs)
            policy = torch.distributions.categorical.Categorical(logits=logit)
            act = policy.sample()
            cmd = torch.tensor([Brain.ACTION[a] for a in act])
        self.unity.set_command(cmd)
        self.batch.store(exp, act)
        
        # Start learning
        print(f'\rBatch: {self.batch.size()}/{self.batch_size_max}', end='')
        if self.batch.size() >= self.batch_size_max:
            loss = self.learning(self.batch.pop())
            print()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self, batch):
        """Training the brain using batch data"""
        for episode in batch:    # Replace rewards with weights
            episode[:, -1] = sum(episode[:, -1]) * torch.ones_like(episode[:, -1])
        data = torch.cat(batch, dim=0)

        # Calculate loss
        logit = self.brain(data[:, 0:40])
        policy = torch.distributions.categorical.Categorical(logits=logit)
        logp = policy.log_prob(data[:, -2])
        weight = data[:, -1]
        loss = -(logp * weight).mean()

        # Optimize the brain
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