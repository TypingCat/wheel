#!/usr/bin/env python3

import rclpy
from rclpy.logging import LoggingSeverity
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

class Simulation(Node):
    """Simulate agents in Unity, and publish samples to ROS"""

    def __init__(self):
        super().__init__('wheel_navigation_simulation')
        self.get_logger().set_level(LoggingSeverity.INFO)

        # Connect with Unity
        self.get_logger().warning("Wait for Unity scene play")
        self.env = UnityEnvironment(file_name=None, seed=1)
        self.env.reset()
        self.get_logger().info("Unity environment connected")

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
        self.brain = Brain(40, 2)
        print(self.brain)

        # Initialize ROS
        self.get_logger().info("Start simulation")
        self.sample_publisher = self.create_publisher(String, '/sample', 10)
        self.brain_state_subscription = self.create_subscription(String, '/brain/update', self.brain_update_callback, 1)
        self.timer = self.create_timer(0.2, self.timer_callback)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass
        self.get_logger().info("Unity environment disconnected")

    def timer_callback(self):
        # Simulate environment one-step forward
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior)

        exp = self.get_experience()
        obs = list(map(lambda agent: exp[agent]['obs'], exp))

        # Get action
        with torch.no_grad():
            act = self.brain(torch.tensor(obs))

        print([decision_steps.agent_id.size, terminal_steps.agent_id.size])

        # Set action
        self.env.set_actions(self.behavior, act.numpy())

        # Publish experience
        sample = self.pack_sample(exp, act.tolist())
        self.sample_publisher.publish(sample)


        print("----")
        for agent in exp:
            print(self.pre_exp[agent]['obs'][0:2])
            print(exp[agent]['obs'][0:2])
            break


        # Backup experience
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

    def get_experience(self, exp={}):
        """Get observation, done, reward from unity environment"""

        # Organize experiences into a dictionary
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        if len(terminal_steps) > 0:
            for agent in terminal_steps:
                try:
                    exp[agent]
                except:
                    exp[agent] = {}
                exp[agent]['obs'] = list(map(float, terminal_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(terminal_steps[agent].reward)
                exp[agent]['done'] = True
        else:
            for agent in decision_steps:
                try:
                    exp[agent]
                except:
                    exp[agent] = {}
                exp[agent]['obs'] = list(map(float, decision_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(decision_steps[agent].reward)
                exp[agent]['done'] = False
                
        return exp

    def pack_sample(self, exp, act):
        # Generate a sample from experiences
        sample = {}
        for agent in exp:
            if self.pre_exp[agent]['done'] is True:
                continue
            sample[str(agent)] = {}
            sample[str(agent)]['obs'] = self.pre_exp[agent]['obs']
            sample[str(agent)]['act'] = act[agent]
            sample[str(agent)]['reward'] = exp[agent]['reward']
            sample[str(agent)]['done'] = exp[agent]['done']
            sample[str(agent)]['next_obs'] = exp[agent]['obs']

        # Convert sample to json
        sample_json = String()
        sample_json.data = json.dumps(sample)
        return sample_json

def main(args=None):
    rclpy.init(args=args)
    node = Simulation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()