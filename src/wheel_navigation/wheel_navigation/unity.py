#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from mlagents_envs.environment import UnityEnvironment


class Unity(Node):
    def __init__(self):
        # Connect with Unity
        self.notice("Wait for Unity scene play")
        self.env = UnityEnvironment(file_name=None, seed=1)
        self.env.reset()
        self.notice("Unity environment connected")

        # Get behavior specification
        for behavior_name in self.env.behavior_specs:
            self.behavior = behavior_name
            break
        spec = self.env.behavior_specs[self.behavior]
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)

        print(f"Behavior name: {self.behavior}")
        print(f"Observation shapes: {spec.observation_shapes}")
        print(f"Action shapes: {spec.action_shape}")
        print(f"Number of agents: {len(decision_steps)+len(terminal_steps)}\n")

        # Initialize ROS
        super().__init__('wheel_navigation_unity')
        self.count = 0
        self.timer = self.create_timer(1, self.timer_callback)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass
        self.notice("Unity environment disconnected")

    def timer_callback(self):
        # for behavior_name in self.env.behavior_specs:
        #     spec = self.env.behavior_specs[behavior_name]
        #     decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        
        self.env.step()

        print(self.count)
        self.count += 1        

    def notice(self, string):
        """Print yellow string"""
        print('\033[93m' + string + '\033[0m')


def main(args=None):
    rclpy.init(args=args)
    node = Unity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()