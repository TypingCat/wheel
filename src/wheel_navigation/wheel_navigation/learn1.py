#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import torch

from wheel_navigation.sim1 import Brain

class Batch:
    """Dataset for SPG"""

    def __init__(self, batch_size, obs_size, act_size):
        self.batch_size = batch_size
        self.obs_size = obs_size
        self.act_size = act_size
        self.reset()
        
    def reset(self):
        self.obs = torch.empty([0, self.obs_size])
        self.act = torch.empty([0, self.act_size])
        self.weight = []
        self.returns = []
        self.lengths = []
        self.is_used = False

    def append(self, obs):
        if self.check_free_space() > 0:
            self.obs = torch.cat([self.obs, obs], dim=0)

    def check_free_space(self):
        return self.batch_size - self.obs.shape[0]

class SPG(Node):
    """Simple Policy Gradient"""

    def __init__(self):
        # Set parameters
        num_input = 40
        num_output = 9

        # Initialize brain
        self.brain = Brain(num_input, num_output)
        self.batch = Batch(batch_size=2000, obs_size=num_input, act_size=num_output)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

        # Initialize ROS
        super().__init__('wheel_navigation_learning')
        self.sample_subscription = self.create_subscription(String, '/sample', self.sample_callback, 1)
        self.brain_update_publisher = self.create_publisher(String, '/brain/update', 10)
        self.timer = self.create_timer(2, self.timer_callback)

    def sample_callback(self, msg):
        # Convert json into tensor
        sample = json.loads(msg.data)
        obs = []
        for agent in sample:
            obs.append(sample[agent]['obs'])
            # velocitiy = sample[agent]['obs'][36:38]
            # target = sample[agent]['obs'][38:40]

            self.batch.append(
                sample[agent]['obs'],
                sample[agent]['act']
            )
            
        obs = torch.tensor(obs).float()

        print("------------")
        for agent in sample:
            print(sample[agent]['obs'][0:2])
            print(sample[agent]['act'])
            print(sample[agent]['reward'])
            print(sample[agent]['done'])
            print(sample[agent]['next_obs'][0:2])
            break        

        # Accumulate data
        self.batch.append(obs)
        # print(self.batch.check_free_space())
        

        # Train the brain
        # act = self.brain(obs)

        # logits = self.brain(obs)
        # for logit in logits:
        #     # idx = torch.distributions.categorical.Categorical(logits=logit).sample().item()
        #     act = torch.cat([act, act_set[idx]], dim=0)
        # logp = torch.distributions.categorical.Categorical(logits=logits).log_prob(act)



        # loss = self.criterion(act, ctrl)        

        # loss = -(logp * weights).mean()


        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # Log
        # self.get_logger().warning(f"Loss {float(loss):.4f}")

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

def main(args=None):
    rclpy.init(args=args)
    node = SPG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()