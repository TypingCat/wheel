#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from mlagents_envs.environment import UnityEnvironment


class Unity(Node):

    def __init__(self):
        super().__init__('wheel_navigation_unity')

        print("Wait for Unity scene play")
        self.env = UnityEnvironment(file_name=None, seed=1)
        # self.env.reset()
        print("Unity environment connected")

        self.timer = self.create_timer(0.1, self.timer_callback)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass
        print("Unity environment disconnected")

    def timer_callback(self):
        print('step')
        self.env.step()


def main(args=None):
    rclpy.init(args=args)
    node = Unity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()