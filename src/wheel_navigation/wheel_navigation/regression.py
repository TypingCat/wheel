#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import math
import numpy as np

import torch
from . import brain

class Regression(Node):
    """Learning action using a supervisor"""

    def __init__(self):
        # Initialize brain
        self.brain = brain.Brain()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

        # Initialize ROS
        super().__init__('wheel_navigation_regression')
        self.sample_subscription = self.create_subscription(String, '/sample', self.sample_callback, 1)
        self.brain_update_publisher = self.create_publisher(String, '/brain/update', 10)
        self.timer = self.create_timer(2, self.timer_callback)

    def sample_callback(self, msg):
        # Convert json into tensor
        sample = json.loads(msg.data)
        obs = []
        ctrl = []
        for agent in sample:
            obs.append(sample[agent]['obs'])
            # velocitiy = sample[agent]['obs'][36:38]
            target = sample[agent]['obs'][38:40]
            ctrl.append(self.supervisor(target))
        obs = torch.tensor(obs).float()
        ctrl = torch.tensor(ctrl).float()

        # Train the brain
        act = self.brain(obs)
        loss = self.criterion(act, ctrl)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        print(loss)

    def timer_callback(self):
        # Extract brain state as list dictionary
        state_tensor = self.brain.state_dict()
        state_list = {}
        for key, value in state_tensor.items():
            state_list[key] = value.tolist()
        
        # Publish brain state
        state = String()
        state.data = json.dumps(state_list)
        self.brain_update_publisher.publish(state)
    
    def supervisor(self, target, max_linear_velocity=0.5, max_angular_velocity=1.0):
        """Simple P-control that prioritizes angular velocity"""

        angular_ctrl = np.sign(target[1]) * min(abs(target[1]), max_angular_velocity)
        linear_weight = math.cos(min(abs(target[1]), 0.5*math.pi))
        linear_ctrl = linear_weight * np.sign(target[0]) * min(abs(target[0]), max_linear_velocity)
        return [linear_ctrl, angular_ctrl]

def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()