#!/usr/bin/env python3

import torch
import time
import rclpy
from rclpy.node import Node

from wheel_navigation.env import Unity

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

class Batch:
    def __init__(self, size):
        self.size = size
        self.reset()
        
    def reset(self):
        self.data = torch.empty(0, 0)

    def check_free_space(self):
        return self.size - self.data.shape[0]

    def extend(self, sample):
        if self.data.shape[0] == 0:
            self.data = torch.empty(0, sample.shape[1])
        if self.check_free_space() > 0:
            self.data = torch.cat([self.data, sample], dim=0)

class SPG(Node):
    """Simple Policy Gradient"""

    def __init__(self):
        super().__init__('wheel_navigation_spg')
        self.brain = Brain(num_input=40, num_output=9)
        self.batch = Batch(size=120)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.02)
        print(self.brain)
        
        self.get_logger().warning("Wait for Unity scene play")
        self.unity = Unity()
        self.get_logger().info("Unity environment connected")

        # Initialize experience
        # exp = {}
        # while(len(exp) == 0):
        #     print(".", end="")
        #     self.env.step()
        #     exp = self.get_experience()
        # self.pre_exp = exp

        # Initialize ROS
        self.get_logger().info("Start simulation")
        self.time_start = time.time()
        self.timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        # Simulate environment one-step forward
        self.unity.env.step()
        exp = self.unity.get_experience()
        if not exp:
            return

        # Set commands
        obs = torch.tensor([exp[agent]['obs'] for agent in exp])
        logits = self.brain(obs)
        policy = torch.distributions.categorical.Categorical(logits=logits)
        act = policy.sample()
        cmd = torch.tensor([Brain.ACTION[a] for a in act])
        self.unity.set_command(cmd)

        # Calculate weights




        # Accumulate samples into batch
        # for agent in exp:
        #     sample[str(agent)] = {}
        #     sample[str(agent)]['agent'] = float(exp[agent]['agent'])
        #     sample[str(agent)]['obs'] = self.pre_exp[agent]['obs']
        #     sample[str(agent)]['reward'] = exp[agent]['reward']
        #     sample[str(agent)]['done'] = exp[agent]['done']
        #     sample[str(agent)]['act'] = act[agent]
        #     sample[str(agent)]['next_obs'] = exp[agent]['obs']
        logp = policy.log_prob(act).unsqueeze(dim=1)
        sample = torch.cat([obs, logp], dim=1)
        self.batch.extend(sample)

        # # Start learning
        # if self.batch.check_free_space() <= 0:
        #     loss = self.learning()
        #     self.get_logger().warning(
        #         f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")
        print("!")

    def learning(self):
        """Training the brain using batch data"""
        obs = self.batch.data[:, 0:40]
        logp = self.batch.data[:, 40:41]

        # Calculate loss
        # logp = get_policy(obs).log_prob(act)
        # loss = -(logp * weights).mean()
        
        # # Optimize the brain
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.batch.reset()
        return 0.

def main(args=None):
    rclpy.init(args=args)
    node = SPG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()