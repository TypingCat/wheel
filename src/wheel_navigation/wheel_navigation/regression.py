#!/usr/bin/env python3

import rclpy

import torch

from wheel_navigation.env import Environment

from rclpy.node import Node
from std_msgs.msg import String

from mlagents_envs.environment import UnityEnvironment
import copy
import json

import torch

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

    def get_command(self, act):
        return act

class Unity():
    def __init__(self):
        self.connect()

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
        agents = list(set(list(decision_steps.agent_id) + list(terminal_steps.agent_id)))
        print(f"Behavior name: {self.behavior}")
        print(f"Observation shapes: {spec.observation_shapes}")
        print(f"Action specifications: {spec.action_spec}")
        print(f"Agents: {agents}")

    def __del__(self):
        try:
            self.env.close()
        except:
            pass

class Regression(Node):
    """Simple regression for test the learning environment"""

    def __init__(self):
        super().__init__('wheel_navigation_regression')
        self.brain = Brain(40, 2)
        print(self.brain)

        self.get_logger().warning("Wait for Unity scene play")
        self.unity = Unity()
        self.get_logger().info("Unity environment connected")

        # # Initialize experience
        # exp = {}
        # while(len(exp) == 0):
        #     print(".", end="")
        #     self.env.step()
        #     exp = self.get_experience()
        # self.pre_exp = exp

        # Initialize ROS
        self.get_logger().info("Start simulation")
        # self.sample_publisher = self.create_publisher(String, '/sample', 10)
        # self.brain_state_subscription = self.create_subscription(String, '/brain/update', self.brain_update_callback, 1)
        self.timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        # Simulate environment one-step forward
        self.env.step()
        exp = self.get_experience()
        obs = list(map(lambda agent: exp[agent]['obs'], exp))
        obs = torch.tensor(obs)

        # Get action
        with torch.no_grad():
            act = self.brain(obs)
            cmd = self.brain.get_command(act)

        # Set action
        try:
            self.env.set_actions(self.behavior, cmd.numpy())
        except:
            return

        # Publish experience
        sample = self.pack_sample(exp, act.tolist())
        self.sample_publisher.publish(sample)
        self.pre_exp = copy.deepcopy(exp)

    def brain_update_callback(self, msg):
        """Syncronize brain using json state"""

        # Convert json to tensor
        state_list = json.loads(msg.data)
        state_tensor = {}
        for key, value in state_list.items():
            state_tensor[key] = torch.tensor(value)

        # Update brain with msg
        self.brain.load_state_dict(state_tensor)

    def get_experience(self):
        """Get observation, done, reward from unity environment"""

        # Organize experiences into a dictionary
        exp = {}
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        for agent in terminal_steps:
            try:
                exp[agent]
            except:
                exp[agent] = {}
                exp[agent]['agent'] = agent
                exp[agent]['obs'] = list(map(float, terminal_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(terminal_steps[agent].reward)
                exp[agent]['done'] = True
        for agent in decision_steps:
            try:
                exp[agent]
            except:
                exp[agent] = {}
                exp[agent]['agent'] = agent
                exp[agent]['obs'] = list(map(float, decision_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(decision_steps[agent].reward)
                exp[agent]['done'] = False

        return exp

    def pack_sample(self, exp, act):
        # Generate a sample from experiences
        sample = {}
        for agent in exp:
            sample[str(agent)] = {}
            sample[str(agent)]['agent'] = float(exp[agent]['agent'])
            sample[str(agent)]['obs'] = self.pre_exp[agent]['obs']
            sample[str(agent)]['reward'] = exp[agent]['reward']
            sample[str(agent)]['done'] = exp[agent]['done']
            sample[str(agent)]['act'] = act[agent]
            sample[str(agent)]['next_obs'] = exp[agent]['obs']

        # Convert sample to json
        sample_json = String()
        sample_json.data = json.dumps(sample)
        return sample_json

class Batch:
    def __init__(self, size):
        self.size = size
        self.reset()
        
    def reset(self):
        self.data = []

    def check_free_space(self):
        return self.size - len(self.data)

    def extend(self, samples):
        if self.check_free_space() > 0:     # Allows oversize
            self.data += samples

def main(args=None):
    rclpy.init(args=args)
    node = Regression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()