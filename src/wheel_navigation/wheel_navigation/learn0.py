#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import math
import numpy as np
import torch
import copy
import threading
import time

from wheel_navigation.sim0 import Brain
from wheel_navigation.env import Batch

class Regression(Node):
    """Learning action using a supervisor"""

    def __init__(self):
        # Initialize brain
        self.brain = Brain(num_input=40, num_output=2)
        self.batch = Batch(size=1)
        self.threads = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)
        self.time = time.time()

        # Initialize ROS
        super().__init__('wheel_navigation_learning')
        self.sample_subscription = self.create_subscription(String, '/sample', self.sample_callback, 1)
        self.brain_update_publisher = self.create_publisher(String, '/brain/update', 10)

    def sample_callback(self, msg):
        # Accumulate samples into batch
        samples = json.loads(msg.data)
        self.batch.extend(
            [samples[agent] for agent in samples])
        print(self.batch.check_free_space())

        # Insert batch into learning thread
        self.threads = [t for t in self.threads if t.isAlive()]
        if self.batch.check_free_space() <= 0:
            t = threading.Thread(
                target=self.learning,
                args=[copy.deepcopy(self.batch)],
                daemon=True)
            self.threads.append(t)
            self.batch.reset()
            t.start()

    def learning(self, batch):
        obs = [samples['obs'] for samples in batch.data]
        target = [o[38:40] for o in obs]

        # Calculate answers of supervisor
        act_answer = [self.supervisor(t) for t in target]
        obs = torch.tensor(obs).float()
        act_answer = torch.tensor(act_answer).float()

        # Train the brain
        act = self.brain(obs)
        loss = self.criterion(act, act_answer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        self.publish_brain_state()
        self.get_logger().warning(f"Time {time.time()-self.time:.1f}s, Loss {float(loss):.4f}")

    def publish_brain_state(self):
        # Extract brain state as list dictionary
        state_tensor = self.brain.state_dict()
        state_list = {}
        for key, value in state_tensor.items():
            state_list[key] = value.tolist()
        
        # Publish brain state
        state = String()
        state.data = json.dumps(state_list)
        self.brain_update_publisher.publish(state)
    
    def supervisor(self, target):
        """Simple P-control that prioritizes angular velocity"""

        angular_ctrl = np.sign(target[1]) * min(abs(target[1]), 1)
        linear_weight = math.cos(min(abs(target[1]), 0.5*math.pi))
        linear_ctrl = linear_weight * np.sign(target[0]) * min(abs(target[0]), 1)
        return [linear_ctrl, angular_ctrl]

def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()