#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mlagents_envs.environment import UnityEnvironment
import copy
import json

import torch
from . import brain

import numpy as np
import matplotlib.pyplot as plt


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
        agents = list(set(list(decision_steps.agent_id) + list(terminal_steps.agent_id)))

        print(f"Behavior name: {self.behavior}")
        print(f"Observation shapes: {spec.observation_shapes}")
        print(f"Action specifications: {spec.action_spec}")
        print(f"Agents: {agents}")

        # Initialize experience
        exp = {}
        while(len(exp) < len(agents)):
            print(".", end="")
            self.env.step()
            exp = self.get_experience(exp)
        self.pre_exp = exp

        # Initialize brain
        self.brain = brain.Brain(
            spec.observation_shapes[0][0],      # Input size
            spec.action_spec.continuous_size)   # Output size

        # Initialize ROS
        super().__init__('wheel_navigation_unity')
        self.notice("start")
        self.sample_publisher = self.create_publisher(String, '/sample', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass
        self.notice("Unity environment disconnected")

    def timer_callback(self):
        # Simulate environment one-step forward
        self.env.step()
        exp = self.get_experience()

        # Convert observation list to tensor
        obs = []
        for agent in exp:
            obs.append(exp[agent]['obs'])
        
        # Decide action
        with torch.no_grad():
            act = self.brain(torch.tensor(obs))
        self.env.set_actions(self.behavior, act.numpy())

        # Publish experience
        sample = self.wrap(exp, act.tolist())
        self.sample_publisher.publish(sample)

        # Backup
        self.pre_exp = exp

    def get_experience(self, exp={}):
        """Get observation, done, reward from unity environment"""

        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        if len(terminal_steps) > 0:
            for agent in terminal_steps:
                try:
                    exp[agent]
                except:
                    exp[agent] = {}
                exp[agent]['obs'] = list(map(float, terminal_steps[agent].obs[0].tolist()))
                exp[agent]['done'] = True

                # self.calc_reward(exp[agent]['obs'])

                exp[agent]['reward'] = float(terminal_steps[agent].reward)
        else:
            for agent in decision_steps:
                try:
                    exp[agent]
                except:
                    exp[agent] = {}
                exp[agent]['obs'] = list(map(float, decision_steps[agent].obs[0].tolist()))
                exp[agent]['done'] = False

                # print(agent)
                # self.calc_reward(exp[agent]['obs'])

                exp[agent]['reward'] = float(decision_steps[agent].reward)
                
        return exp

    def calc_reward(self, obs):

        scan = obs[:1000]
        vel = obs[1000:1002]
        target = obs[1002:1004]

        # print(['vel', vel])
        # print(['target', target])        


    def wrap(self, exp, act):
        # Generate a sample from experiences
        sample = {}
        for agent in exp:
            if self.pre_exp[agent]['done'] is True:
                continue
            sample[str(agent)] = copy.deepcopy(self.pre_exp[agent])
            sample[str(agent)]['action'] = act[agent]
            sample[str(agent)]['next_obs'] = exp[agent]['obs']

        # Convert sample to json
        sample_json = String()
        sample_json.data = json.dumps(sample)
        return sample_json

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