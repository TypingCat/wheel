#!/usr/bin/env python3

import rclpy

import torch

from rclpy.node import Node

from mlagents_envs.environment import UnityEnvironment

import torch
import numpy as np
import math
import time

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

    def get_command(act):
        return act

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

class Unity():
    def __init__(self):
        self.connect()

    def __del__(self):
        try:
            self.env.close()
        except:
            pass

    def connect(self):
        # Connect with Unity
        self.env = UnityEnvironment(file_name=None, seed=1)
        self.env.reset()

        # Get behavior specification
        for behavior_name in self.env.behavior_specs:
            self.behavior = behavior_name
            break
        spec = self.env.behavior_specs[self.behavior]
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        self.agents = list(set(list(decision_steps.agent_id) + list(terminal_steps.agent_id)))

        print(f"Behavior name: {self.behavior}")
        print(f"Observation shapes: {spec.observation_shapes}")
        print(f"Action specifications: {spec.action_spec}")
        print(f"Agents: {self.agents}")

    def get_experience(self):
        """Get observation, done, reward from unity environment"""
        exp = {}
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)

        if len(set(decision_steps) - set(terminal_steps)) == len(self.agents):
            for agent in self.agents:
                exp[agent] = {}
            for agent in terminal_steps:
                exp[agent]['obs'] = list(map(float, terminal_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(terminal_steps[agent].reward)
                exp[agent]['done'] = True
            for agent in list(set(decision_steps) - set(terminal_steps)):
                exp[agent]['obs'] = list(map(float, decision_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(decision_steps[agent].reward)
                exp[agent]['done'] = False
        return exp
    
    def set_command(self, cmd):
        self.env.set_actions(self.behavior, cmd.detach().numpy())

class Regression(Node):
    """Simple regression for test the learning environment"""

    def __init__(self):
        super().__init__('wheel_navigation_regression')
        self.brain = Brain(num_input=40, num_output=2)
        self.batch = Batch(size=1)
        self.criterion = torch.nn.MSELoss()
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

        # Set actions
        obs = torch.tensor([exp[agent]['obs'] for agent in exp.keys()])
        act = self.brain(obs)
        cmd = Brain.get_command(act)
        self.unity.set_command(cmd)
        
        # Accumulate samples into batch
        # for agent in exp:
        #     sample[str(agent)] = {}
        #     sample[str(agent)]['agent'] = float(exp[agent]['agent'])
        #     sample[str(agent)]['obs'] = self.pre_exp[agent]['obs']
        #     sample[str(agent)]['reward'] = exp[agent]['reward']
        #     sample[str(agent)]['done'] = exp[agent]['done']
        #     sample[str(agent)]['act'] = act[agent]
        #     sample[str(agent)]['next_obs'] = exp[agent]['obs']
        sample = torch.cat([obs, act], dim=1)
        self.batch.extend(sample)

        # Start learning
        if self.batch.check_free_space() <= 0:
            loss = self.learning()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self):
        """Training the brain using batch data"""
        target = self.batch.data[:, 38:40]
        act = self.batch.data[:, 40:42]

        # Calculate loss
        advice = Brain.supervisor(target)
        loss = self.criterion(act, advice)
        
        # Optimize the brain
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.batch.reset()
        return loss

def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()