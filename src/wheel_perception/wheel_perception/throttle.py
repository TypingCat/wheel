#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sensor_msgs.msg
from cv_bridge import CvBridge


class Throttle(Node):

    def __init__(self):
        super().__init__('wheel_vision_throttle')
        self.bridge = CvBridge()
        self.image = sensor_msgs.msg.CompressedImage()

        self.publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage, 'camera/color/image_raw/compressed', 1)
        self.subscription = self.create_subscription(
            sensor_msgs.msg.Image, '/camera/color/image_raw', self.topic_callback, 1)
        self.timer = self.create_timer(0.067, self.timer_callback)  # 15Hz
        
    def topic_callback(self, msg):
        self.image = msg

    def timer_callback(self):
        image_cv2 = self.bridge.imgmsg_to_cv2(self.image)
        image_compressed = self.bridge.cv2_to_compressed_imgmsg(image_cv2)
        self.publisher.publish(image_compressed)


def main(args=None):
    rclpy.init(args=args)
    node = Throttle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()