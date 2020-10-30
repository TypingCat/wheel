#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from my_msgs.msg import Add


class Unity(Node):

    def __init__(self):
        super().__init__('unity')
        print('Hi Unity!')
        

def main(args=None):
    rclpy.init(args=args)
    node = Unity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()