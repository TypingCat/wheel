#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# from mlagents_envs.environment import UnityEnvironment
# import copy
import json

# import torch
from . import brain


class Regression(Node):
    """Learning actions directly from experiences"""

    def __init__(self):
        # # Initialize brain
        # self.brain = brain.Brain(
        #     spec.observation_shapes[0][0],      # Input size
        #     spec.action_spec.continuous_size)   # Output size

        # Initialize ROS
        super().__init__('wheel_navigation_unity')
        self.sample_subscription = self.create_subscription(String, '/sample', self.sample_callback, 1)

    def sample_callback(self, msg):

        # Subscribe json into a dictionary
        sample = json.loads(msg.data)
        for agent in sample:
            print(agent)
            # print(sample[agent]['obs'])
            print(sample[agent]['done'])
            print(sample[agent]['reward'])
            print(sample[agent]['action'])
            # print(sample[agent]['next_obs'])


        



def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()