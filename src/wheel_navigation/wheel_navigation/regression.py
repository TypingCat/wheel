#!/usr/bin/env python3

import torch
import numpy as np
import math
import time

import rclpy
from rclpy.node import Node

from wheel_navigation.env import Unity

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

    def supervisor(target):
        """Simple P-control that prioritizes angular velocity"""
        distance = target[:, 0].tolist()
        angle = target[:, 1].tolist()
        
        angular_ctrl = [np.sign(a)*min(abs(a), 1) for a in angle]
        linear_weight = [math.cos(min(abs(a), 0.5*math.pi)) for a in angle]
        linear_ctrl = [w*np.sign(d)*min(abs(d), 1) for w, d in zip(linear_weight, distance)]
        
        return torch.cat([
            torch.tensor(linear_ctrl).unsqueeze(1),
            torch.tensor(angular_ctrl).unsqueeze(1)], dim=1).float()
            
class Regression(Node):
    """Simple regression for test the learning environment"""

    def __init__(self):
        super().__init__('wheel_navigation_regression')
        self.brain = Brain(num_input=40, num_output=2)
        self.batch = []
        self.batch_size_max = 60
        self.criterion = torch.nn.MSELoss()
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
        act = self.brain(obs)
        self.unity.set_command(act)
        self.batch.append(torch.cat([obs, act], dim=1))

        # Start learning
        batch_size = sum([b.shape[0] for b in self.batch])
        print(f'\rBatch: {batch_size}/{self.batch_size_max}', end='')
        if batch_size >= self.batch_size_max:
            loss = self.learning(self.batch)
            self.batch = []
            print()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self, batch):
        """Training the brain using batch data"""
        data = torch.cat(batch, dim=0)

        # Calculate loss
        target = data[:, 38:40]
        act = data[:, 40:42]
        advice = Brain.supervisor(target)
        loss = self.criterion(act, advice)
        
        # Optimize the brain
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()